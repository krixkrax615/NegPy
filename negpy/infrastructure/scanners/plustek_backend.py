# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for the pyopticfilm Plustek USB driver."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import (
    MAX_N_PASSES,
    MIN_N_PASSES,
    MultiExposureMode,
    ScanMode,
    ScanParams,
)
from pyopticfilm.asic.gl128 import DEFAULT_IMAGE_USB_PACE_S
from pyopticfilm.device.select import model_for_device, model_is_scan_ready
from pyopticfilm.exceptions import (
    DeviceNotFoundError,
    DriverBindingError,
    PlustekError,
    ScanCancelled,
    UsbError,
)
from pyopticfilm.logging import get_logger
from pyopticfilm.scanner import Scanner
from pyopticfilm.usb.device import UsbDeviceInfo, find_devices, list_devices
from negpy.infrastructure.scanners.result import ScanResult

logger = get_logger(__name__)


def _ensure_quiet_usb_drain(scanner: Scanner) -> None:
    asic = scanner.asic
    if hasattr(asic, "image_usb_pace_s"):
        asic.image_usb_pace_s = DEFAULT_IMAGE_USB_PACE_S


#: Auto-exposure meters each frame with a quick scan at this dpi before the real one.
_METER_DPI = 600

#: Samples = 2 scans at twice the dpi and averages each 2x2 block back down. The optics resolve no
#: more at the higher dpi, so the average keeps the detail and collects four sensor reads per pixel.
_SAMPLE_BIN = 2


# Register 0x31 reads 0xFE at rest; pressing either of the two buttons pulls bit 3 or bit 6 low.
_BUTTON_REGISTER = 0x31
_BUTTON_MASK = 0x48


def _bin_samples(arr: np.ndarray) -> np.ndarray:
    """Average each 2x2 block of an HxW or HxWxC uint16 image (odd edge row/column dropped)."""
    a = np.asarray(arr)
    h, w = a.shape[0] // _SAMPLE_BIN * _SAMPLE_BIN, a.shape[1] // _SAMPLE_BIN * _SAMPLE_BIN
    out = np.zeros((h // _SAMPLE_BIN, w // _SAMPLE_BIN, *a.shape[2:]), dtype=np.uint32)
    for dy in range(_SAMPLE_BIN):
        for dx in range(_SAMPLE_BIN):
            out += a[dy:h:_SAMPLE_BIN, dx:w:_SAMPLE_BIN]
    n = _SAMPLE_BIN * _SAMPLE_BIN
    return ((out + n // 2) // n).astype(a.dtype)


#: A metered scan whose film peak lands below this fraction of the peak metering predicted is treated as a glitch.
_BLACK_SCAN_RATIO = 0.15


def scan_looks_black(rgb: np.ndarray, expected_peak: float) -> bool:
    """True when the film in ``rgb`` is far darker than metering predicted (a blank or black scan).

    Open light and its edge are dropped, as metering does, then the brightest channel's 99.9th percentile of the
    rest is compared with ``expected_peak`` (counts of 65535). Too little film to judge counts as fine.
    """
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    inner = a[h // 20 : h - h // 20 : 8, w // 20 : w - w // 20 : 8].astype(np.float32)
    open_light = (inner >= 0.9 * 65535.0).any(axis=2).astype(np.uint8)
    # The edge of open light is soft: grow the mask by about 1.5 % of the frame so it is not read as film.
    grown = cv2.dilate(open_light, np.ones((7, 7), np.uint8)).astype(bool)
    film = inner[~grown]
    if film.shape[0] < 1000 or expected_peak <= 0:
        return False
    return float(np.percentile(film, 99.9, axis=0).max()) < _BLACK_SCAN_RATIO * expected_peak


def _caps_for(model: Any) -> ScannerCapabilities:
    from pyopticfilm.scan.bringup import PRESCAN_DPI, default_frame_crop_norm, is_gl128_opticfilm

    # Prescan and Multi-Pass are GL128 session features. Other scan-ready leaves (e.g. 7600i
    # v1) report scan_ready but raise on those paths.
    gl128 = is_gl128_opticfilm(model)
    return ScannerCapabilities(
        ir_channel=model.supports_infrared,
        supported_dpi=tuple(sorted(model.resolutions_dpi)),
        supported_depths=model.bpp_color,
        sources=(ScanMode.TRANSPARENCY,),
        max_area_mm=model.max_area_mm,
        auto_exposure=gl128,
        autofocus=False,
        prescan=gl128,
        prescan_dpi=PRESCAN_DPI if gl128 else 0,
        prescan_mirror_x=bool(getattr(model, "mirror_x", False)) if gl128 else False,
        prescan_default_crop=default_frame_crop_norm(model) if gl128 else None,
        multi_exposure=bool(getattr(model, "scan_ready", False) and getattr(model, "exposure_long", None)),
        # Multi-Pass needs no long-exposure register; it is GL128-only in pyopticfilm.
        max_n_passes=MAX_N_PASSES if gl128 else MIN_N_PASSES,
        max_samples=_SAMPLE_BIN if gl128 else 1,
        raw_capture=gl128,
        button_scan=gl128,
        adapter_frame_capacity=None,
        adapter_frame_control=False,
        can_eject=False,
        frame_pitch_mm=0.0,
        exposure_time_us=None,
    )


def _to_scanner_device(info: UsbDeviceInfo) -> ScannerDevice:
    model = model_for_device(info.product_id, getattr(info, "bcd_device", 0))
    return ScannerDevice(
        id=info.device_id,
        vendor=model.vendor,
        model=model.model,
        capabilities=_caps_for(model),
    )


def _safe_progress(
    progress: Callable[[float, str], None] | None,
    value: float,
    phase: str = "Scanning",
) -> None:
    if progress is None:
        return
    with suppress(Exception):
        progress(max(0.0, min(1.0, float(value))), phase)


def _gl128_me_pass_layout(*, capture_ir: bool, multi_exposure: bool, n_passes: int = 1) -> tuple[int, int] | None:
    """(n_early, n_pass): physical passes before, and total physical passes across, the single
    short→long exposure-change boundary. Every repeat within a slot shares one exposure — no
    pyopticfilm-side "preparing" moment between repeats — so there is exactly one boundary
    regardless of ``n_passes``: all short-slot repeats (+ the optional IR pass) happen first,
    then all long-slot repeats. Collapses to the pre-Multi-Pass ``(2,3)``/``(1,2)`` layout at
    ``n_passes=1``."""
    if not multi_exposure:
        return None
    ir_extra = 1 if capture_ir else 0
    n_early = ir_extra + n_passes
    return n_early, n_early + n_passes


def _make_scan_progress(
    progress: Callable[[float, str], None] | None,
    *,
    multi_exposure: bool,
    capture_ir: bool,
    n_passes: int = 1,
) -> Callable[[float], None]:
    layout = _gl128_me_pass_layout(capture_ir=capture_ir, multi_exposure=multi_exposure, n_passes=n_passes)
    if layout is None:

        def scan_progress(p: float) -> None:
            _safe_progress(progress, 0.1 + 0.9 * p)

        return scan_progress

    n_early, n_pass = layout
    plateau = n_early / n_pass
    state = {"long_started": False}

    def scan_progress(p: float) -> None:
        frac = min(1.0, max(0.0, float(p)))
        if frac >= 1.0 - 1e-9:
            _safe_progress(progress, 0.85, "Merging exposures")
            return
        if frac > plateau + 1e-6:
            state["long_started"] = True
            _safe_progress(progress, 0.10 + 0.72 * frac, "Scanning")
        elif abs(frac - plateau) < 1e-6 and not state["long_started"]:
            _safe_progress(progress, 0.83, "Preparing long exposure")
        else:
            _safe_progress(progress, 0.10 + 0.72 * frac, "Scanning")

    return scan_progress


def _capture_info(params: ScanParams, *, dpi: int, film_exposure: int | None) -> str:
    """One line saying how a scan was captured, for the file's ImageDescription."""
    parts = [f"{dpi} dpi", f"{params.depth}-bit"]
    if params.multi_exposure_mode != MultiExposureMode.OFF:
        parts.append("adaptive multi-exposure")
    if params.n_passes > 1:
        parts.append(f"multi-pass x{params.n_passes}")
    if params.samples > 1:
        parts.append(f"multi-sampling x{params.samples}")
    if film_exposure is not None:
        parts.append(f"auto-exposure {film_exposure}")
    parts.append("raw (no driver sharpening)" if params.raw_capture else "driver capture sharpening")
    return "NegPy Plustek scan: " + ", ".join(parts)


def _validate_params(params: ScanParams, *, model: Any | None = None) -> None:
    from pyopticfilm.device.model_8200i import MODEL_8200I

    dpi = int(params.dpi)
    depth = int(params.depth)
    resolutions = getattr(model, "resolutions_dpi", None) if model is not None else None
    bpp = getattr(model, "bpp_color", None) if model is not None else None
    if not isinstance(resolutions, (tuple, list)) or not resolutions:
        resolutions = MODEL_8200I.resolutions_dpi
    if not isinstance(bpp, (tuple, list)) or not bpp:
        bpp = MODEL_8200I.bpp_color
    if dpi not in resolutions:
        raise RuntimeError(f"Unsupported dpi={dpi}; supported={sorted(resolutions)}")
    if depth not in bpp:
        raise RuntimeError(f"Unsupported depth={depth}; supported={sorted(bpp)}")

    if params.frame is not None:
        raise RuntimeError(f"Frame {params.frame} requested but the device has no frame-selection option")
    if params.auto_exposure:
        from pyopticfilm.scan.bringup import is_gl128_opticfilm

        if model is None or not is_gl128_opticfilm(model):
            raise RuntimeError("Auto-exposure requested but the device has no 'ae' option")
    if params.autofocus:
        raise RuntimeError("Autofocus requested but the device has no autofocus option")
    if params.capture_ir and model is not None and getattr(model, "supports_infrared", None) is False:
        raise RuntimeError(f"{getattr(model, 'model', 'device')} does not support infrared")
    if params.capture_ir and model is not None:
        ir_resolutions = getattr(model, "infrared_resolutions_dpi", None)
        if isinstance(ir_resolutions, (tuple, list)) and ir_resolutions and dpi not in ir_resolutions:
            raise RuntimeError(
                f"{getattr(model, 'model', 'device')} scans infrared at {', '.join(str(d) for d in sorted(ir_resolutions))} dpi"
            )
    mode = params.multi_exposure_mode
    if mode != MultiExposureMode.OFF and model is not None and not getattr(model, "exposure_long", None):
        raise RuntimeError(f"{getattr(model, 'model', 'device')} does not support multi-exposure")
    if not (MIN_N_PASSES <= params.n_passes <= MAX_N_PASSES):
        raise RuntimeError(f"n_passes={params.n_passes} out of range ({MIN_N_PASSES}-{MAX_N_PASSES})")
    if params.n_passes > 1:
        from pyopticfilm.scan.bringup import is_gl128_opticfilm

        if model is not None and not is_gl128_opticfilm(model):
            raise RuntimeError(f"{getattr(model, 'model', 'device')} does not support Multi-Pass")
        if params.capture_ir:
            raise RuntimeError("IR and Multi-Pass cannot be combined yet — scan IR separately, or set Passes to 1.")
    if params.samples > 1:
        from pyopticfilm.scan.bringup import is_gl128_opticfilm

        name = getattr(model, "model", "device")
        if params.samples != _SAMPLE_BIN or (model is not None and not is_gl128_opticfilm(model)):
            raise RuntimeError(f"{name} supports Samples 1 or {_SAMPLE_BIN}")
        resolutions = tuple(getattr(model, "resolutions_dpi", ()) or ())
        if resolutions and dpi * _SAMPLE_BIN not in resolutions:
            usable = ", ".join(str(d) for d in sorted(resolutions) if d * _SAMPLE_BIN in resolutions)
            raise RuntimeError(f"Samples {_SAMPLE_BIN} scans at twice the dpi; use one of {usable} dpi")


class PlustekSession:
    """Exclusive device hold for batch workflows."""

    def __init__(
        self,
        backend: PlustekBackend,
        device_id: str,
        scanner: Scanner,
    ) -> None:
        self.device_id = device_id
        self._backend = backend
        self._scanner = scanner
        self._closed = False

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        return self._backend._scan_on_scanner(self._scanner, params, progress, cancel)

    def eject(self) -> bool:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._scanner.close()
        finally:
            self._backend._release_session(self)

    def __enter__(self) -> PlustekSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class PlustekBackend:
    """ScannerBackend for Plustek OpticFilm scanners (pyopticfilm)."""

    def __init__(self, *, calib_cache: Path | None = None) -> None:
        try:
            import usb.core  # noqa: F401
        except ImportError as exc:
            raise ScannerUnavailable(
                "Plustek USB needs pyopticfilm (uv sync --group plustek). "
                "On Windows, bind the scanner's USB id to WinUSB with Zadig (see docs/PLUSTEK_WINDOWS.md) — "
                "e.g. 07b3:1825 (8200i SE), 07b3:1824 (8100 V2), 07b3:181f (8300i SE), or 07b3:0c3b (7600i v1)."
            ) from exc
        self._calib_cache = calib_cache
        self._sessions: dict[str, PlustekSession] = {}
        # One open of the USB device at a time: a scan holds it, a button poll only borrows it between scans.
        self._usb_lock = threading.Lock()

    def list_devices(self) -> list[ScannerDevice]:
        try:
            devices = [_to_scanner_device(d) for d in find_devices(supported_only=True)]
        except DeviceNotFoundError:
            return []
        except DriverBindingError as exc:
            raise ScannerUnavailable(str(exc)) from exc
        return [d for d in devices if d.capabilities.sources]

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def open_session(self, device_id: str) -> ScannerSession:
        if device_id in self._sessions:
            raise RuntimeError(f"Device already held in a session: {device_id}")
        self._ensure_known_device(device_id)
        try:
            with self._usb_lock:
                scanner = Scanner.open(device_id, calib_cache=self._calib_cache)
                if not scanner.asic._initialized:
                    scanner.asic.init()
                _ensure_quiet_usb_drain(scanner)
                if not scanner.asic.is_at_home():
                    scanner.home()
        except DriverBindingError as exc:
            raise ScannerUnavailable(str(exc)) from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except DeviceNotFoundError as exc:
            raise RuntimeError(f"Unknown or disconnected device: {device_id}") from exc

        session = PlustekSession(self, device_id, scanner)
        self._sessions[device_id] = session
        return session

    def _release_session(self, session: PlustekSession) -> None:
        self._sessions.pop(session.device_id, None)

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if device_id in self._sessions:
            raise RuntimeError(f"Device {device_id} is held by an open session; use session.scan()")
        self._ensure_known_device(device_id)
        try:
            with self._usb_lock, Scanner.open(device_id, calib_cache=self._calib_cache) as scanner:
                if not scanner.asic._initialized:
                    scanner.asic.init()
                _ensure_quiet_usb_drain(scanner)
                if not scanner.asic.is_at_home():
                    scanner.home()
                return self._scan_on_scanner(scanner, params, progress, cancel)
        except TransientScanError:
            raise
        except ScannerUnavailable:
            raise
        except DriverBindingError as exc:
            raise TransientScanError(str(exc)) from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except DeviceNotFoundError as exc:
            raise RuntimeError(f"Unknown or disconnected device: {device_id}") from exc

    def eject(self, device_id: str) -> bool:
        del device_id
        return False

    def poll_button(self, device_id: str) -> bool:
        """True while a front-panel button is held. Opens the device for one register read; False when a
        scan or session has it, or the read fails."""
        if device_id in self._sessions or not self._usb_lock.acquire(blocking=False):
            return False
        try:
            with Scanner.open(device_id, calib_cache=self._calib_cache) as scanner:
                raw = scanner.asic.protocol.read_register(_BUTTON_REGISTER)
            return bool(~raw & _BUTTON_MASK)
        except Exception:  # noqa: BLE001 - a failed poll only means "not pressed"
            return False
        finally:
            self._usb_lock.release()

    def _ensure_known_device(self, device_id: str) -> None:
        ids = {d.device_id for d in find_devices(supported_only=True)}
        if device_id in ids:
            return
        known = {d.device_id: d for d in list_devices()}
        if device_id in known:
            raise RuntimeError(
                f"{device_id} is not a supported Plustek film scanner "
                "(e.g. 07b3:1825 8200i SE, 07b3:1824 8100 V2, 07b3:181f 8300i SE, 07b3:0c3b 7600i v1)."
            )
        raise RuntimeError(f"Unknown or disconnected device: {device_id}")

    def _scan_on_scanner(
        self,
        scanner: Scanner,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if not model_is_scan_ready(scanner.model):
            msg = (
                f"{scanner.model.model} ({scanner.model.asic}) cannot scan with "
                "pyOpticfilm in this release — only scan-ready pyopticfilm models are supported. "
                "See the pyopticfilm release notes for the current list."
            )
            if sys.platform != "win32":
                msg += " Try Backend → SANE if that backend lists this scanner."
            raise RuntimeError(msg)
        _validate_params(params, model=scanner.model)
        dpi = int(params.dpi)
        capture_ir = bool(params.capture_ir)
        me_mode = params.multi_exposure_mode
        multi_exposure = me_mode != MultiExposureMode.OFF
        n_passes = int(params.n_passes)
        window = params.window
        sampled = int(params.samples) > 1
        scan_dpi = dpi * _SAMPLE_BIN if sampled else dpi
        geometry = self._default_scan_geometry(scanner, dpi=scan_dpi, window=window)

        _safe_progress(progress, 0.0)
        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")

        try:
            self._ensure_colour_calib(
                scanner,
                dpi=scan_dpi,
                window=window,
                geometry=geometry,
                progress=progress,
                cancel=cancel,
            )

            film_exposure = None
            if params.auto_exposure and not multi_exposure:
                # Multi-Exposure meters each frame itself, so Auto-exposure only drives single exposures.
                meter_geometry = self._default_scan_geometry(scanner, dpi=_METER_DPI, window=window)
                self._ensure_colour_calib(
                    scanner,
                    dpi=_METER_DPI,
                    window=window,
                    geometry=meter_geometry,
                    progress=progress,
                    cancel=cancel,
                )
                _safe_progress(progress, 0.05, "Metering")
                film_exposure = scanner.meter_film_exposure(
                    resolution=_METER_DPI,
                    area=None if meter_geometry is not None else window,
                    geometry=meter_geometry,
                    cancel=cancel,
                )
                self._ensure_colour_calib(
                    scanner,
                    dpi=scan_dpi,
                    window=window,
                    geometry=geometry,
                    progress=progress,
                    cancel=cancel,
                )

            scan_progress = _make_scan_progress(
                progress,
                multi_exposure=multi_exposure,
                capture_ir=capture_ir,
                n_passes=n_passes,
            )

            def on_status(status: str) -> None:
                if status == "priming":
                    _safe_progress(progress, 0.05, "Priming")
                elif status == "scanning":
                    _safe_progress(progress, 0.1, "Scanning")
                # "prime_skipped" when gl128_prime is off (default on both GL128 models).

            scan_area = None if geometry is not None else window
            expected_peak = None
            meter_peak = getattr(scanner, "last_meter_peak", None)
            if film_exposure is not None and meter_peak:
                expected_peak = min(65535.0, meter_peak * film_exposure / float(scanner.model.exposure_short))

            for attempt in (1, 2):
                rgb_image = scanner.scan(
                    resolution=scan_dpi,
                    mode="color",
                    area=scan_area,
                    geometry=geometry,
                    progress=scan_progress,
                    cancel=cancel,
                    on_status=on_status,
                    multi_exposure=multi_exposure,
                    infrared=capture_ir,
                    align_passes=True,
                    n_passes=n_passes,
                    single_pass_exposure=film_exposure,
                    raw=bool(params.raw_capture),
                )
                if expected_peak is None or not scan_looks_black(np.asarray(rgb_image.rgb), expected_peak):
                    break
                logger.warning("Scan came out far darker than metering predicted (attempt %d of 2)", attempt)
                if attempt == 2:
                    raise TransientScanError(
                        "The scan came out black twice although metering saw the film. "
                        "Power-cycle the scanner and scan again."
                    )
            ir_plane = np.asarray(rgb_image.ir) if capture_ir and rgb_image.ir is not None else None
        except ScanCancelled as exc:
            raise RuntimeError("Scan cancelled") from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except PlustekError as exc:
            raise RuntimeError(str(exc)) from exc

        rgb = np.asarray(rgb_image.rgb)
        if sampled:
            rgb = _bin_samples(rgb)
            ir_plane = _bin_samples(ir_plane) if ir_plane is not None else None
        if not multi_exposure:
            _safe_progress(progress, 1.0)
        return ScanResult(
            rgb=rgb,
            ir=ir_plane,
            dpi=dpi,
            device_model=rgb_image.device_model or f"{scanner.model.vendor} {scanner.model.model}",
            ir_valid_mask=None,
            capture_info=_capture_info(params, dpi=dpi, film_exposure=film_exposure),
        )

    @staticmethod
    def _ensure_colour_calib(
        scanner: Scanner,
        *,
        dpi: int,
        window: tuple[float, float, float, float] | None,
        geometry: object | None,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> None:
        """Home ASIC shading before the colour feed (apply cache or measure).

        Progress uses 0–0.1 so a cold measure does not look hung. Film may stay
        loaded — shading runs at home before position feeds, like SilverFast.
        GL128 only: other scan-ready leaves (e.g. 7600i v1) calibrate live in-scan.
        """
        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")
        _safe_progress(progress, 0.02, "Calibrating")
        from pyopticfilm.exceptions import CalibrationError
        from pyopticfilm.scan.bringup import bringup_scan_geometry, is_gl128_opticfilm
        from pyopticfilm.scan.geometry import ScanGeometry

        if not is_gl128_opticfilm(scanner.model):
            _safe_progress(progress, 0.1, "Calibrating")
            return

        geo = geometry if isinstance(geometry, ScanGeometry) else None
        if geo is None and window is not None:
            from pyopticfilm.scan.geometry import compute_geometry

            geo = compute_geometry(dpi, model=scanner.model, area=window)
        if geo is None:
            geo, _ = bringup_scan_geometry(scanner.model, dpi, profile="preview_safe")
        if geo is None:
            _safe_progress(progress, 0.1, "Calibrating")
            return

        asic = scanner.asic
        calibrator = scanner.calibrator
        hit = calibrator.find_for_scan(method="transparency", geometry=geo)
        if hit is not None and hit.has_asic_blob and getattr(asic, "asic_shading_ready", False):
            _safe_progress(progress, 0.1, "Calibrating")
            return

        logger.info("Ensuring colour ASIC shading for dpi=%d", dpi)
        try:
            # Prefer ensure (apply or measure); Scanner.calibrate also parks motor.
            was_armed = scanner._bringup_motor_armed
            try:
                if getattr(scanner.model, "asic", "") == "GL128":
                    scanner.disarm_bringup_motor()
                calibrator.ensure_colour_asic_shading(geo)
            finally:
                if was_armed:
                    scanner.arm_bringup_motor()
        except CalibrationError as exc:
            raise RuntimeError(str(exc)) from exc
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        _safe_progress(progress, 0.1, "Calibrating")

    @staticmethod
    def _default_scan_geometry(
        scanner: Scanner,
        *,
        dpi: int,
        window: tuple[float, float, float, float] | None,
    ) -> object | None:
        """Lab Full-window or crop geometry for GL128 scan-ready models (forced into scan)."""
        from pyopticfilm.scan.bringup import (
            bringup_scan_geometry,
            crop_scan_geometry,
            is_gl128_opticfilm,
        )

        if not is_gl128_opticfilm(scanner.model):
            return None
        if window is not None:
            geometry, _meta = crop_scan_geometry(scanner.model, dpi, window)
            return geometry
        geometry, _meta = bringup_scan_geometry(scanner.model, dpi, profile="preview_safe")
        return geometry

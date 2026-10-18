"""Per-card eye toggle: a transient A/B view of the edit with one or more cards at their defaults.

The eye never touches the stored edit. ``effective_config`` derives the config the preview renders
from the saved one plus the set of cards switched off; nothing here is persisted, pushed to history,
exported, or copied. Geometry is left out on purpose: its fields (crop rect, rotation, flips) decide the
frame's size and the ROI everything else is metered on, so a "default" Geometry is a different picture, not
an A/B of one setting. The Roll-tab cards drive roll defaults, not this frame's look, so they have no eye.
"""

from __future__ import annotations

from dataclasses import replace
from typing import AbstractSet, Optional

from negpy.desktop.settings_catalog import COLOR_FIELDS, DENOISE_FIELDS, LAB_FIELDS, PAPER_FIELDS, TONE_FIELDS
from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcessConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import auto_meter_for_mode, cast_removal_for_mode
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

# Card key (the ControlsPanel section key) -> (WorkspaceConfig section, owned fields or None for the whole section).
EYE_CARDS: dict[str, tuple[str, Optional[tuple[str, ...]]]] = {
    "color": ("exposure", COLOR_FIELDS),
    "tone": ("exposure", TONE_FIELDS),
    "paper": ("exposure", PAPER_FIELDS),
    "lab": ("lab", LAB_FIELDS),
    "denoise": ("lab", DENOISE_FIELDS),
    "altproc": ("altproc", None),
    "toning": ("toning", None),
    "retouch": ("retouch", None),
    "finish": ("finish", None),
}

EYE_LABELS = {
    "color": "Filtration",
    "tone": "Tone",
    "paper": "Paper Response",
    "lab": "Lab",
    "denoise": "Denoise",
    "altproc": "Alternative Processes",
    "toning": "Toning",
    "retouch": "Retouch",
    "finish": "Finishing",
}

_DEFAULT_SECTIONS = {
    "lab": LabConfig(),
    "altproc": AltProcessConfig(),
    "toning": ToningConfig(),
    "retouch": RetouchConfig(),
    "finish": FinishConfig(),
}
_DEFAULT_EXPOSURE = DEFAULT_WORKSPACE_CONFIG.exposure
_AUTO_METER_FIELDS = ("auto_exposure", "auto_normalize_contrast")


def default_exposure_field(field: str, process_mode: str):
    """The value *field* defaults to on a frame in *process_mode*. Auto Density/Auto Grade and Cast Removal
    default per mode; every other ExposureConfig field has one flat default."""
    default = getattr(_DEFAULT_EXPOSURE, field)
    if field in _AUTO_METER_FIELDS:
        return auto_meter_for_mode(process_mode, default)
    if field == "cast_removal_strength":
        return cast_removal_for_mode(process_mode, default)
    return default


def _card_defaults(card: str, process_mode: str) -> dict:
    section, fields = EYE_CARDS[card]
    if section == "exposure":
        return {f: default_exposure_field(f, process_mode) for f in fields or ()}
    base = _DEFAULT_SECTIONS[section]
    if fields is None:
        return {name: getattr(base, name) for name in base.__dataclass_fields__}
    return {f: getattr(base, f) for f in fields}


def effective_config(cfg: WorkspaceConfig, disabled: AbstractSet[str]) -> WorkspaceConfig:
    """*cfg* with every card in *disabled* at its defaults. Returns *cfg* itself when nothing applies, so
    identity checks against the saved config keep working."""
    cards = [c for c in EYE_CARDS if c in disabled]
    if not cards:
        return cfg
    out = cfg
    for card in cards:
        section = EYE_CARDS[card][0]
        sub = getattr(out, section)
        out = replace(out, **{section: replace(sub, **_card_defaults(card, cfg.process.process_mode))})
    return out


def card_edited(a: WorkspaceConfig, b: WorkspaceConfig, card: str) -> bool:
    """True when *card*'s own fields differ between two configs."""
    section = EYE_CARDS[card][0]
    names = tuple(_card_defaults(card, a.process.process_mode))
    sa, sb = getattr(a, section), getattr(b, section)
    return any(getattr(sa, n) != getattr(sb, n) for n in names)


def disabled_key(disabled: AbstractSet[str]) -> tuple[str, ...]:
    """Stable identity of a disabled set, for cache keys."""
    return tuple(c for c in EYE_CARDS if c in disabled)

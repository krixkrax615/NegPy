from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, section_subheader
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import ProcessMode


class DenoiseSidebar(BaseSidebar):
    """
    Noise removal after the print is decoded. Chroma Denoise is a luma-guided filter on CIELAB a*/b*: colour
    speckle goes, luminance grain stays. Luma Denoise is an edge-preserving smoothing of L*. Their values live on
    LabConfig (chroma_denoise, luma_denoise) so saved edits keep working.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.lab

        self.color_noise_header = section_subheader("COLOR NOISE")
        self.layout.addWidget(self.color_noise_header)

        self.chroma_denoise_slider = CompactSlider("Chroma Denoise", 0.0, 5.0, conf.chroma_denoise)
        self.layout.addWidget(self.chroma_denoise_slider)

        self.chroma_hint = hint_label(
            "Removes color speckle by letting color follow the picture's brightness; grain stays as luminance. 1-2 for scans, 3+ for very noisy ones.",
            "muted",
        )
        self.layout.addWidget(self.chroma_hint)

        self.luma_noise_header = section_subheader("LUMINANCE NOISE")
        self.layout.addWidget(self.luma_noise_header)

        self.luma_denoise_slider = CompactSlider("Luma Denoise", 0.0, 1.0, conf.luma_denoise)
        self.layout.addWidget(self.luma_denoise_slider)

        self.luma_hint = hint_label(
            "Smooths grain in flat areas and keeps edges and texture. Light touch first: 0.2-0.4 softens grain, above 0.7 starts to look painted.",
            "muted",
        )
        self.layout.addWidget(self.luma_hint)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.chroma_denoise_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, chroma_denoise=v)
        )
        self.chroma_denoise_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, chroma_denoise=v)
        )
        self.luma_denoise_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, luma_denoise=v)
        )
        self.luma_denoise_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, luma_denoise=v)
        )

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.chroma_denoise_slider.setValue(conf.chroma_denoise)
            self.luma_denoise_slider.setValue(conf.luma_denoise)
            # Color noise only exists in color: hide the group with its slider in B&W.
            self.color_noise_header.setVisible(not is_bw)
            self.chroma_denoise_slider.setVisible(not is_bw)
            self.chroma_hint.setVisible(not is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        self.chroma_denoise_slider.blockSignals(blocked)
        self.luma_denoise_slider.blockSignals(blocked)

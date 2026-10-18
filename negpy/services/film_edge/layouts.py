"""Maker edge print layouts, in roll millimeters. A new maker style is one `LAYOUTS` entry."""

from dataclasses import dataclass
from enum import StrEnum


class Band(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class EdgeRun:
    """One repeating line of edge print.

    `on`: "number" at each frame number's label, "half" at each A label (135 only), or "pitch" every `pitch` mm from
    `at`. `text` is a template over {n} {label} {maker} {text} {code}; an item that renders empty is not printed.
    `lead`, `trail` and `under` name a mark: "triangle", "outline", "arrow" or "dots"."""

    on: str
    band: Band
    text: str
    cap: float
    offset: float = 1.0
    at: float = 0.0
    pitch: float = 0.0
    count: int = 0  # 0: to the roll end
    first: int = 1  # {n} of the first "pitch" item
    align: str = "center"
    max_width: float = 0.0
    lead: str = ""
    trail: str = ""
    under: str = ""


@dataclass(frozen=True)
class EdgeLayout:
    runs_135: tuple[EdgeRun, ...]
    runs_120: tuple[EdgeRun, ...]
    # Palettes ("color", "bw") whose 135 edge carries the DX code; slides never do.
    dx: tuple[str, ...] = ("color",)


TOP, BOTTOM = Band.TOP, Band.BOTTOM
# 120 numbers run at the maker's own pitch, not in step with the camera's frames.
_CAP_120, _OFFSET_120 = 1.8, 1.3


def _numbers_120(lead: str) -> tuple[EdgeRun, ...]:
    return (
        EdgeRun("pitch", TOP, "{maker} {text}", _CAP_120, _OFFSET_120, at=30.0, pitch=86.0, count=19, max_width=60.0),
        EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=43.0, count=19, lead=lead),
    )


_KODAK_TOP_135 = (
    EdgeRun("number", TOP, "{n}", 1.2),
    EdgeRun("number", TOP, "{maker} {text}", 1.25, at=15.5, max_width=26.0),
)

LAYOUTS: dict[str, EdgeLayout] = {
    "kodak": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.4),
            EdgeRun("half", BOTTOM, "{label}", 0.9, offset=1.4, under="arrow"),
            *_KODAK_TOP_135,
        ),
        runs_120=(
            EdgeRun("pitch", TOP, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=44.5, count=15, first=41),
            EdgeRun("pitch", TOP, "{maker} {text}", _CAP_120, _OFFSET_120, at=30.25, pitch=89.0, count=8, max_width=36.0),
            EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=20.0, pitch=60.0, count=12, trail="triangle"),
        ),
    ),
    "fuji": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.5),
            EdgeRun("half", BOTTOM, "{label}", 1.5, lead="arrow"),
            EdgeRun("number", TOP, "{n}", 1.5),
            EdgeRun("number", TOP, "{code}", 1.0, at=9.5, max_width=7.0),
            EdgeRun("number", TOP, "{n}A", 1.5, at=19.0),
            EdgeRun("number", TOP, "FUJI", 1.0, at=28.5),
        ),
        runs_120=(
            EdgeRun("pitch", TOP, "FUJI {code}", _CAP_120, _OFFSET_120, at=30.0, pitch=84.0, count=19, max_width=50.0),
            EdgeRun("pitch", BOTTOM, "{code}", _CAP_120, _OFFSET_120, at=6.4, pitch=42.0, count=19, align="right", max_width=24.0),
            EdgeRun("pitch", BOTTOM, "{n}", _CAP_120, _OFFSET_120, at=8.0, pitch=42.0, count=19, align="left", lead="outline"),
        ),
    ),
    "ilford": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.5),
            EdgeRun("half", BOTTOM, "{label}", 1.1, lead="triangle"),
            EdgeRun("pitch", TOP, "{maker}", 1.6, at=12.0, pitch=96.0),
            EdgeRun("pitch", TOP, "{text}", 1.6, at=60.0, pitch=96.0, max_width=40.0),
        ),
        runs_120=_numbers_120("dots"),
        dx=("color", "bw"),
    ),
    "cine": EdgeLayout(
        runs_135=(EdgeRun("pitch", TOP, "{maker} {text}", 1.2, at=12.0, pitch=76.0, max_width=60.0),),
        runs_120=_numbers_120("triangle"),
        dx=(),
    ),
    "generic": EdgeLayout(
        runs_135=(
            EdgeRun("number", BOTTOM, "{label}", 1.4),
            EdgeRun("half", BOTTOM, "{label}", 1.1, lead="triangle"),
            *_KODAK_TOP_135,
        ),
        runs_120=_numbers_120("triangle"),
    ),
}

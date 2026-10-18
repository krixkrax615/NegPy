"""Film edge print, in roll millimeters. `offset` is the distance from the film edge to an item's center."""

import math
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

from negpy.features.metadata.gear_models import FilmEdge, FilmStock
from negpy.services.film_edge.geometry import HALF_FRAME_PITCH, FilmGeometry, SheetFormat
from negpy.services.film_edge.layouts import LAYOUTS, Band, EdgeRun

LABEL_PITCH_135 = HALF_FRAME_PITCH
DX_MODULE = 0.406
DX_MODULES = 31
DX_START = 1.9
DX_DATA_TRACK = (0.0, 1.0)
DX_CLOCK_TRACK = (1.0, 2.2)
# A DX label ends this far past its position, clear of the barcode's quiet zone.
DX_LABEL_END = 1.5
# Longer than any item's half-width, so an item that straddles a cut is kept and clipped.
REACH = 45.0
GENERIC = "generic"


@dataclass(frozen=True)
class EdgeText:
    """`lead`, `trail` and `under` name a mark, as in `EdgeRun`."""

    x: float
    band: Band
    text: str
    cap: float
    offset: float
    align: str = "center"
    max_width: float = 0.0
    lead: str = ""
    trail: str = ""
    under: str = ""


@dataclass(frozen=True)
class EdgeCode:
    """Bits read left to right; 1 prints light."""

    x: float
    clock: tuple[int, ...]
    data: tuple[int, ...]

    @property
    def length(self) -> float:
        return DX_MODULES * DX_MODULE


EdgeItem = Union[EdgeText, EdgeCode]

_CINE_WORDS = ("vision", "cinestill", "double-x", "eastman", "eterna", "5207", "5219", "5222", "5203", "5213", "7222")
_ILFORD_MAKERS = ("ILFORD", "KENTMERE", "HARMAN")
_FUJI_MAKERS = ("FUJIFILM", "FUJICOLOR", "FUJICHROME", "FUJI")
_MAKER_WORDS = ("KODAK", "EASTMAN", *_ILFORD_MAKERS, *_FUJI_MAKERS)


def ascii_upper(text: str) -> str:
    """Folds accents so every glyph exists in the edge face."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(folded.upper().split())


def edge_family(stock: str, manufacturer: str = "") -> str:
    """The `LAYOUTS` key guessed from a film's name, for a stock with no edge data."""
    text = f"{manufacturer} {stock}".lower()
    if any(word in text for word in _CINE_WORDS):
        return "cine"
    if "kodak" in text:
        return "kodak"
    if any(maker.lower() in text for maker in _ILFORD_MAKERS):
        return "ilford"
    if "fuji" in text:
        return "fuji"
    return GENERIC


def split_maker(name: str, manufacturer: str = "") -> tuple[str, str]:
    """(maker, product) of a film name in edge capitals; a leading maker word comes off the name."""
    maker = ascii_upper(manufacturer)
    words = ascii_upper(name).split()
    if words and (words[0] == maker or words[0] in _MAKER_WORDS):
        return words[0], " ".join(words[1:])
    return maker, " ".join(words)


@dataclass(frozen=True)
class EdgeStyle:
    layout: str = GENERIC
    maker: str = ""
    text: str = ""
    # A short mark (Fuji's RVP); the layout prints `text` where it is empty.
    code: str = ""
    dx: bool = False
    dx_number: Optional[tuple[int, int]] = None
    # False: rebate and perforations only.
    printed: bool = True

    @property
    def dx_product(self) -> tuple[int, int]:
        """The stock's DX number, else a stand-in from its name; never 0, which readers reject."""
        if self.dx_number is not None:
            return self.dx_number
        digest = zlib.crc32(f"{self.maker} {self.text}".strip().encode("ascii", "ignore"))
        return 1 + digest % 127, (digest >> 8) % 16


def parse_dx(text: str) -> Optional[tuple[int, int]]:
    """(part1, part2) of "95-13"; None unless both fit the edge code."""
    part1, sep, part2 = text.partition("-")
    if not (sep and part1.strip().isdigit() and part2.strip().isdigit()):
        return None
    number = int(part1), int(part2)
    return number if 1 <= number[0] <= 127 and 0 <= number[1] <= 15 else None


def edge_style(stock: Optional[FilmStock], library: Any = None, name: str = "", manufacturer: str = "") -> EdgeStyle:
    """Edge print of a catalog stock; with no stock, guessed from a free-text film name and maker."""
    if stock is None:
        maker, text = split_maker(name, manufacturer)
        return EdgeStyle(edge_family(name, manufacturer), maker, text)
    owner = _edge_owner(stock, library)
    edge = owner.edge or FilmEdge()
    maker, text = split_maker(owner.stock_name, owner.manufacturer)
    layout = edge.style or edge_family(owner.stock_name, owner.manufacturer)
    return EdgeStyle(
        layout if layout in LAYOUTS else GENERIC,
        edge.maker or maker,
        edge.text or text,
        edge.code,
        dx_number=parse_dx(edge.dx),
    )


def _edge_owner(stock: FilmStock, library: Any) -> FilmStock:
    """The stock whose edge data applies: a personal copy without its own reads the bundled stock of the same name."""
    if stock.edge is None:
        key = (stock.manufacturer.casefold(), stock.stock_name.casefold())
        for twin in getattr(library, "film_stocks", ()):
            if twin.is_bundled and twin.edge is not None and (twin.manufacturer.casefold(), twin.stock_name.casefold()) == key:
                stock = twin
                break
    if stock.edge is not None and stock.edge.prints_as and library is not None:
        source = library.get_film_stock(stock.edge.prints_as)
        if source is not None:
            return source
    return stock


def label_text(index: int) -> str:
    return f"{index // 2 + 1}{'A' if index % 2 else ''}"


def dx_bits(number: int, half: bool, product: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(clock, data) tracks of a 31-module DX film edge barcode (ISO 1007, zxing layout)."""
    p1, p2 = product

    def bits(value: int, count: int) -> list[int]:
        return [(value >> shift) & 1 for shift in range(count - 1, -1, -1)]

    payload = [0, *bits(p1, 7), 0, *bits(p2, 4), *bits(number % 64, 6), int(half), 0]
    parity = sum(payload) % 2
    data = (1, 0, 1, 0, 1, *payload, parity, 0, 1, 0, 1)
    clock = (1, 1, 1, 1, 1, *[i % 2 for i in range(23)], 1, 1, 1)
    return clock, data


def edge_items(
    geometry: FilmGeometry,
    style: EdgeStyle,
    frame_count: int,
    x0: float,
    x1: float,
    numbers: Optional[Sequence[int]] = None,
) -> list[EdgeItem]:
    """Items that can touch x0..x1; the strip clips the ones that straddle a cut.

    `numbers` is each frame slot's place on the roll; None means 0, 1, 2 …"""
    if not style.printed:
        return []
    layout = LAYOUTS.get(style.layout, LAYOUTS[GENERIC])
    fields = {"maker": style.maker, "text": style.text, "code": style.code or style.text}
    items: list[EdgeItem] = []
    if geometry.format == SheetFormat.MEDIUM:
        runs = layout.runs_120
    else:
        runs = layout.runs_135
        dx = style.dx and bool(layout.dx)
        slots = list(numbers) if numbers is not None else list(range(frame_count))
        for x, j in _label_slots(geometry, slots[:frame_count]):
            number, is_a = j // 2 + 1, bool(j % 2)
            for run in runs:
                if run.on != ("half" if is_a else "number"):
                    continue
                anchor, align = x + run.at, run.align
                if dx and run.band == Band.BOTTOM:
                    anchor, align = anchor + DX_LABEL_END, "right"
                _add_text(items, run, anchor, align, {**fields, "n": number, "label": label_text(j)})
            if dx:
                clock, data = dx_bits(number, is_a, style.dx_product)
                items.append(EdgeCode(x + DX_START, clock, data))
    roll_end = frame_count * geometry.pitch
    for run in runs:
        if run.on == "pitch":
            for k, x in _pitch_positions(run, roll_end):
                _add_text(items, run, x, run.align, {**fields, "n": run.first + k, "label": ""})
    return [item for item in items if x0 - REACH <= item.x <= x1 + REACH]


def _add_text(items: list[EdgeItem], run: EdgeRun, x: float, align: str, fields: dict) -> None:
    text = " ".join(run.text.format(**fields).split())
    if text:
        items.append(EdgeText(x, run.band, text, run.cap, run.offset, align, run.max_width, run.lead, run.trail, run.under))


def _pitch_positions(run: EdgeRun, roll_end: float) -> Iterator[tuple[int, float]]:
    k = 0
    while k < (run.count or math.inf):
        x = run.at + run.pitch * k
        if x > roll_end + run.pitch:
            return
        yield k, x
        k += 1


def _label_slots(geometry: FilmGeometry, slots: Sequence[int]) -> list[tuple[float, int]]:
    """(roll position, label index) of every half-frame label."""
    if geometry.format == SheetFormat.HALF_FRAME:
        return [(geometry.frame_center(i), roll_index) for i, roll_index in enumerate(slots)]
    labels: list[tuple[float, int]] = []
    for i, roll_index in enumerate(slots):
        x = geometry.frame_center(i)
        labels += [(x, 2 * roll_index), (x + LABEL_PITCH_135, 2 * roll_index + 1)]
    return labels

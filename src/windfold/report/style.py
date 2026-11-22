"""the figures' style from report.toml: Inter at print sizes, one color per model family
(windfold and the physics vivid, learned models muted, everything else neutral), markers and
line styles by learner and the conventions every figure shares: panel letters, the zero line,
estimates with their 95% intervals, labeled rows and neutral direct labels."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, NamedTuple, get_args

import matplotlib as mpl
import numpy as np
from cmcrameri import cm as crameri
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.typing import RcKeyType
from pydantic import Field, model_validator

from windfold.config import Frozen, ModelSpec, build, config_root, read_toml

MM_PER_INCH = 25.4
CONFIRMATION = ", confirmation period"
LEARNERS = ("linear", "lookup", "gbm", "graph", "automl")
REPRESENTATIONS: tuple[str, ...] = get_args(ModelSpec.model_fields["representation"].annotation)
HEX = r"^#[0-9a-f]{6}$"
type Hex = Annotated[str, Field(pattern=HEX)]


class Page(Frozen):
    """pad_mm is the room constrained layout keeps around every panel, space the share of a panel
    kept between panels and clearance_mm the least room the check allows between two texts and
    between a text and a panel's edge or the canvas edge. text_width_mm is the width of the
    paper's text: a wider figure is printed scaled down to it, so its type is drawn larger by
    the same factor and prints at the set size."""

    widths_mm: tuple[float, ...]
    text_width_mm: float = Field(gt=0)
    max_height_mm: float = Field(gt=0)
    max_margin_mm: float = Field(gt=0)
    pad_mm: float = Field(ge=0)
    space: float = Field(ge=0)
    clearance_mm: float = Field(ge=0)

    def print_scale(self, width_mm: float) -> float:
        """the factor a figure this wide is scaled by when printed in the paper."""
        return min(1.0, self.text_width_mm / width_mm)


class Type(Frozen):
    family: str
    files: tuple[str, ...]
    size_pt: float
    min_pt: float
    panel_letter_pt: float
    panel_letter_weight: int

    @model_validator(mode="after")
    def readable(self) -> "Type":
        if min(self.size_pt, self.panel_letter_pt) < self.min_pt:
            raise ValueError(f"type sizes must be at least min_pt ({self.min_pt} pt)")
        return self


class Lines(Frozen):
    data_pt: float
    axes_pt: float
    min_pt: float
    grid_pt: float
    interval_pt: float
    thin_pt: float
    thick_pt: float
    tick_length_pt: float
    marker_pt: float

    @model_validator(mode="after")
    def visible(self) -> "Lines":
        widths = (self.data_pt, self.axes_pt, self.grid_pt, self.interval_pt, self.thin_pt)
        if min(*widths, self.thick_pt) < self.min_pt:
            raise ValueError(f"line widths must be at least min_pt ({self.min_pt} pt)")
        return self


class Neutrals(Frozen):
    """context is for lines of unnamed models drawn behind the named ones: lighter than the
    data, darker than the grid."""

    text: Hex
    axes: Hex
    grid: Hex
    context: Hex
    band: Hex


class Periods(Frozen):
    calibration: Hex
    development: Hex
    confirmatory: Hex


class Files(Frozen):
    raster_dpi: int = Field(gt=0)


class Tables(Frozen):
    significant_figures: int = Field(gt=0)


class Learner(Frozen):
    marker: str
    line: str
    shade: int = Field(ge=0)


class Maps(Frozen):
    magnitude: str
    signed: str
    direction: str


class Farms(Frozen):
    order: tuple[str, ...]
    names: dict[str, str]

    @model_validator(mode="after")
    def named(self) -> "Farms":
        if set(self.order) != set(self.names):
            raise ValueError("farms.order and farms.names must list the same farms")
        return self


class Idea(Frozen):
    farm: str
    history_days: int = Field(gt=0)
    direction_bin_deg: float = Field(gt=0)
    relation_bin_d: float = Field(gt=0)


class Failure(Frozen):
    models: tuple[str, ...]
    min_rows: int = Field(gt=0)


class Monitoring(Frozen):
    models: tuple[str, ...]
    drop: float = Field(gt=0, lt=1)


class Margin(Frozen):
    models: tuple[str, ...] = Field(min_length=1, max_length=4)


class Override(Frozen):
    """a model's own marker, line or shade where its learner's would match another model's, and
    the paper's name for it where that differs from its display name."""

    marker: str | None = None
    line: str | None = None
    shade: int | None = None
    label: str | None = None


class Report(Frozen):
    page: Page
    type: Type
    lines: Lines
    neutrals: Neutrals
    periods: Periods
    files: Files
    tables: Tables
    colours: dict[str, tuple[Hex, ...]]
    learners: dict[str, Learner]
    maps: Maps
    farms: Farms
    regimes: dict[str, str]
    idea: Idea
    turbines: dict[str, str]
    steps: dict[int, str]
    inputs: dict[str, str]
    knows: dict[str, str]
    hypotheses: dict[str, str]
    sensitivities: dict[str, str]
    failure: Failure
    monitoring: Monitoring
    margin: Margin
    overrides: dict[str, Override] = {}

    @model_validator(mode="after")
    def complete(self) -> "Report":
        if set(self.colours) != set(REPRESENTATIONS):
            raise ValueError(f"colours must name exactly {list(REPRESENTATIONS)}")
        if not all(self.colours.values()):
            raise ValueError("each representation needs at least one colour")
        if set(self.learners) != set(LEARNERS):
            raise ValueError(f"learners must name exactly {list(LEARNERS)}")
        if set(self.knows) != set(REPRESENTATIONS):
            raise ValueError(f"knows must name exactly {list(REPRESENTATIONS)}")
        return self


def wrap(name: str, width: int = 18) -> str:
    """a name longer than width on two lines, broken at the space nearest its middle."""
    if len(name) <= width or " " not in name:
        return name
    spaces = [k for k, c in enumerate(name) if c == " "]
    cut = min(spaces, key=lambda k: abs(k - len(name) / 2))
    return name[:cut] + "\n" + name[cut + 1 :]


def load_report(folder: Path | None = None) -> Report:
    return build(Report, read_toml(config_root(folder), "report.toml"), "report.toml")


def learner_of(spec: ModelSpec) -> str:
    """the learner a model's marker and line style stand for; engine models are linear, and the
    one with a per-turbine direction table is the lookup."""
    if spec.learner is not None:
        return spec.learner
    return "lookup" if "turbine_direction_table" in spec.blocks else "linear"


def register_fonts(files: tuple[str, ...]) -> None:
    known = {Path(entry.fname).name for entry in font_manager.fontManager.ttflist}
    folder = resources.files("windfold.report") / "fonts"
    for name in files:
        if name in known:
            continue
        with resources.as_file(folder / name) as path:
            if not path.is_file():
                raise FileNotFoundError(f"the figure font {name} is missing from {folder}")
            font_manager.fontManager.addfont(str(path))


class Dataset(NamedTuple):
    """one farm of one study, as the figures name it."""

    study: str
    farm: str
    label: str


class Style:
    """the report's look; draw every figure inside style.context() so it applies."""

    def __init__(self, report: Report):
        self.report = report
        register_fonts(report.type.files)

    @property
    def rc(self) -> dict[RcKeyType, Any]:
        r = self.report
        t, lines, n = r.type, r.lines, r.neutrals
        pad = r.page.pad_mm / MM_PER_INCH
        return {
            "font.family": t.family,
            **self.type_rc(1.0),
            "axes.titleweight": t.panel_letter_weight,
            "axes.titlelocation": "left",
            "figure.labelweight": "normal",
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": t.family,
            "mathtext.default": "regular",
            "axes.unicode_minus": True,
            "text.color": n.text,
            "axes.labelcolor": n.text,
            "axes.edgecolor": n.axes,
            "xtick.color": n.axes,
            "ytick.color": n.axes,
            "xtick.labelcolor": n.text,
            "ytick.labelcolor": n.text,
            "axes.linewidth": lines.axes_pt,
            "xtick.major.width": lines.axes_pt,
            "ytick.major.width": lines.axes_pt,
            "xtick.minor.width": lines.axes_pt,
            "ytick.minor.width": lines.axes_pt,
            "xtick.major.size": lines.tick_length_pt,
            "ytick.major.size": lines.tick_length_pt,
            "xtick.minor.size": lines.tick_length_pt / 2,
            "ytick.minor.size": lines.tick_length_pt / 2,
            "xtick.major.pad": lines.tick_length_pt,
            "ytick.major.pad": lines.tick_length_pt,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": n.grid,
            "grid.linewidth": lines.grid_pt,
            "lines.linewidth": lines.data_pt,
            "lines.markersize": lines.marker_pt,
            "lines.markeredgewidth": lines.min_pt,
            "patch.linewidth": lines.axes_pt,
            "hatch.linewidth": lines.min_pt,
            "errorbar.capsize": 0,
            "polaraxes.grid": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.constrained_layout.w_pad": pad,
            "figure.constrained_layout.h_pad": pad,
            "figure.constrained_layout.wspace": r.page.space,
            "figure.constrained_layout.hspace": r.page.space,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": r.files.raster_dpi,
        }

    def type_rc(self, scale: float) -> dict[RcKeyType, Any]:
        t = self.report.type
        size, letter = scale * t.size_pt, scale * t.panel_letter_pt
        return {
            "font.size": size,
            "axes.labelsize": size,
            "axes.titlesize": letter,
            "axes.titlepad": 0.8 * size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "legend.fontsize": size,
            "legend.title_fontsize": size,
            "figure.labelsize": size,
        }

    @contextmanager
    def context(self) -> Iterator[None]:
        with mpl.rc_context(self.rc):
            yield

    def figure(self, width_mm: float, height_mm: float) -> Figure:
        """a canvas inside context(); its type is set to print at the report's sizes once the
        figure is scaled to the paper's text width."""
        page = self.report.page
        if width_mm not in page.widths_mm:
            raise ValueError(f"figures are {page.widths_mm} mm wide, not {width_mm}")
        mpl.rcParams.update(self.type_rc(1.0 / page.print_scale(width_mm)))
        return Figure(
            figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), layout="constrained"
        )

    @property
    def size_pt(self) -> float:
        """the body type of the figure being drawn, in points on its canvas."""
        return float(mpl.rcParams["font.size"])

    @property
    def letter_pt(self) -> float:
        """the panel letters of the figure being drawn, in points on its canvas."""
        return float(mpl.rcParams["axes.titlesize"])

    def override(self, spec: ModelSpec) -> Override:
        return self.report.overrides.get(spec.name, Override())

    def colour(self, spec: ModelSpec, shade: int | None = None) -> str:
        shades = self.report.colours[spec.representation]
        own = self.override(spec).shade
        default = self.report.learners[learner_of(spec)].shade if own is None else own
        index = default if shade is None else shade
        return shades[min(index, len(shades) - 1)]

    def representation_colour(self, representation: str, shade: int = 0) -> str:
        shades = self.report.colours[representation]
        return shades[min(shade, len(shades) - 1)]

    def role(self, name: str) -> str:
        """the color of a role: a model family, a neutral or a study period."""
        report = self.report
        if name in report.colours:
            return report.colours[name][0]
        for group in (report.neutrals, report.periods):
            if name in type(group).model_fields:
                return getattr(group, name)
        raise ValueError(f"no colour role {name}")

    def label(self, spec: ModelSpec) -> str:
        return self.override(spec).label or spec.display_name

    def marker(self, spec: ModelSpec) -> str:
        return self.override(spec).marker or self.report.learners[learner_of(spec)].marker

    def line(self, spec: ModelSpec) -> str:
        return self.override(spec).line or self.report.learners[learner_of(spec)].line

    def map(self, kind: str) -> Colormap:
        name = getattr(self.report.maps, kind)
        return getattr(crameri, name)

    def panel_letter(self, ax: Axes, letter: str, title: str | None = None) -> None:
        ax.set_title(letter.lower(), loc="left")
        if title:
            ax.set_title(title, loc="center", fontweight="normal")

    def zero_line(self, ax: Axes, vertical: bool = True) -> None:
        """the line at zero that every difference is read against."""
        draw = ax.axvline if vertical else ax.axhline
        draw(0.0, color=self.report.neutrals.axes, lw=self.report.lines.axes_pt, zorder=1)

    def estimate(
        self,
        ax: Axes,
        at: float,
        centre: float,
        low: float,
        high: float,
        colour: str,
        filled: bool = True,
        marker: str = "o",
    ) -> None:
        """a point estimate on row at with its 95% interval, open when not filled; bounds that
        are not finite draw no interval."""
        lines = self.report.lines
        if np.isfinite([low, high]).all():
            ax.plot([low, high], [at, at], color=colour, lw=lines.interval_pt, zorder=2)
        ax.plot(
            centre,
            at,
            marker=marker,
            color=colour,
            mfc=colour if filled else "white",
            mew=lines.interval_pt,
            ls="none",
            zorder=3,
        )

    def rows(self, ax: Axes, labels: Sequence[str] | None, count: int) -> None:
        """count rows from the top, named on this axes only when labels are given, the names
        kept clear of the panel by the check's clearance."""
        ax.set_ylim(count - 0.5, -0.5)
        ax.set_yticks(range(count), list(labels) if labels is not None else [""] * count)
        clear = 1.5 * self.report.page.clearance_mm / MM_PER_INCH * 72
        ax.tick_params(axis="y", length=0, pad=clear)
        ax.spines["left"].set_visible(False)

    def note(
        self,
        ax: Axes,
        x: float,
        y: float,
        text: str,
        dx: float = 0.0,
        dy: float = 0.0,
        **options: Any,
    ) -> Annotation:
        """a direct label in the text color, dx and dy points away from the data point x, y."""
        return ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            color=self.report.neutrals.text,
            **options,
        )

    def datasets(
        self, farms: Mapping[str, Sequence[str]], confirmatory: Collection[str]
    ) -> list[Dataset]:
        """each study's farms, studies in the mapping's order and farms in the report's; a farm
        of a confirmatory study is named for the confirmation period."""
        return [
            Dataset(
                study, farm, self.farm_name(farm) + (CONFIRMATION if study in confirmatory else "")
            )
            for study, held in farms.items()
            for farm in self.report.farms.order
            if farm in held
        ]

    def farm_name(self, farm: str) -> str:
        return self.report.farms.names.get(farm, farm)

    def regime_name(self, regime: str) -> str:
        return self.report.regimes.get(regime, regime)


def load_style(folder: Path | None = None) -> Style:
    return Style(load_report(folder))

"""draw the diagrams into a folder: python -m windfold.report.diagrams OUT [NAMES]. each is a
vector pdf, an svg for web pages on a solid white background, so it stays readable in dark mode,
and 600 and 1200 pixel previews, written only after the figure check and the spacing check pass.
windfold and wind_frame read the prepared layout; the others need only the configs."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from matplotlib.figure import Figure

from windfold import config
from windfold.report.diagrams import prediction, splits, wind_frame, windfold
from windfold.report.diagrams.output import write
from windfold.report.style import Style, load_style


@dataclass(frozen=True)
class Inputs:
    data: Path
    configs: Path | None


DIAGRAMS: dict[str, Callable[[Inputs, Style], Figure]] = {
    "windfold": lambda inputs, style: windfold.figure(inputs.data, inputs.configs, style),
    "wind_frame": lambda inputs, style: wind_frame.figure(inputs.data, inputs.configs, style),
    "prediction": lambda inputs, style: prediction.figure(style, inputs.configs),
    "splits": lambda inputs, style: splits.figure(style, inputs.configs),
}


def main(
    out: Annotated[Path, typer.Argument(help="folder for the diagrams")],
    names: Annotated[
        list[str] | None, typer.Argument(help=f"diagrams to draw; default all of {list(DIAGRAMS)}")
    ] = None,
    data: Annotated[Path | None, typer.Option(help="data folder; default from resources")] = None,
    configs: Annotated[
        Path | None, typer.Option(help="configs folder; default the packaged one")
    ] = None,
) -> None:
    unknown = sorted(set(names or ()) - set(DIAGRAMS))
    if unknown:
        raise typer.BadParameter(f"unknown diagrams {unknown}; known: {list(DIAGRAMS)}")
    settings = config.load_resources(configs)
    inputs = Inputs(
        data=(data or Path(settings.data_dir)).resolve(),
        configs=configs.resolve() if configs else None,
    )
    style = load_style(configs)
    for name in names or DIAGRAMS:
        for path in write(DIAGRAMS[name](inputs, style), name, out, style):
            typer.echo(path)


if __name__ == "__main__":
    typer.run(main)

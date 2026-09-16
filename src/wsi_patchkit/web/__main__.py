"""Command-line entry point for the bundled WSI viewer."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

SLIDE_EXTENSIONS = frozenset(
    {
        ".bif",
        ".mrxs",
        ".ndpi",
        ".qptiff",
        ".scn",
        ".svs",
        ".tif",
        ".tiff",
        ".vms",
        ".vmu",
    }
)


def _slide(value: str) -> tuple[str, Path]:
    slide_id, separator, path = value.partition("=")
    if not separator or not slide_id or not path:
        raise argparse.ArgumentTypeError("slides must use ID=/path/to/slide.svs")
    return slide_id, Path(path)


def _unique_slide_id(relative_path: Path, existing: set[str]) -> str:
    """Create a stable, URL-safe ID from a path relative to its scan root."""
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.as_posix())
    candidate = candidate.strip("._-") or "slide"
    candidate = candidate[:128]
    if candidate not in existing:
        return candidate

    sequence = 2
    while True:
        suffix = f"-{sequence}"
        unique = f"{candidate[: 128 - len(suffix)]}{suffix}"
        if unique not in existing:
            return unique
        sequence += 1


def discover_slides(
    directories: Iterable[str | Path],
    *,
    existing: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Recursively discover supported WSI files below one or more directories."""
    slides = dict(existing or {})
    ids = set(slides)
    for value in directories:
        directory = Path(value).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths = sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SLIDE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(directory).as_posix().casefold(),
        )
        for path in paths:
            slide_id = _unique_slide_id(path.relative_to(directory), ids)
            slides[slide_id] = path
            ids.add(slide_id)
    return slides


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the wsi-patchkit viewer")
    parser.add_argument(
        "--slide",
        action="append",
        type=_slide,
        metavar="ID=PATH",
        help="register a public slide ID; repeat to expose multiple slides",
    )
    parser.add_argument(
        "--slide-dir",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="recursively register supported WSI files in a directory; repeatable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--tile-size", default=256, type=int)
    parser.add_argument("--reader-pool-size", default=4, type=int)
    parser.add_argument(
        "--crop-output-dir",
        default=Path("crops"),
        type=Path,
        metavar="PATH",
        help="save level-0 rectangular crops in this server-side directory",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install the viewer with: uv sync --extra web") from error

    from .app import create_app

    explicit_slides = dict(args.slide or [])
    try:
        slides = discover_slides(args.slide_dir, existing=explicit_slides)
    except NotADirectoryError as error:
        parser.error(f"slide directory does not exist: {error}")
    if not slides:
        parser.error(
            "provide at least one --slide or a --slide-dir containing WSI files"
        )

    app = create_app(
        slides,
        tile_size=args.tile_size,
        reader_pool_size=args.reader_pool_size,
        crop_output_dir=args.crop_output_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

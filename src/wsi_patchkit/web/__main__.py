"""Command-line entry point for the bundled WSI viewer."""

from __future__ import annotations

import argparse
from pathlib import Path


def _slide(value: str) -> tuple[str, Path]:
    slide_id, separator, path = value.partition("=")
    if not separator or not slide_id or not path:
        raise argparse.ArgumentTypeError("slides must use ID=/path/to/slide.svs")
    return slide_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the wsi-patchkit viewer")
    parser.add_argument(
        "--slide",
        action="append",
        required=True,
        type=_slide,
        metavar="ID=PATH",
        help="register a public slide ID; repeat to expose multiple slides",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--tile-size", default=256, type=int)
    parser.add_argument("--reader-pool-size", default=4, type=int)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install the viewer with: uv sync --extra web") from error

    from .app import create_app

    app = create_app(
        dict(args.slide),
        tile_size=args.tile_size,
        reader_pool_size=args.reader_pool_size,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

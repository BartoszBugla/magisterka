from __future__ import annotations

from collections.abc import Callable

import svgwrite

_PathSeg = tuple[str, str | tuple[float, ...]]


def _add_path(dwg: svgwrite.Drawing, d: str) -> None:
    dwg.add(
        dwg.path(
            d=d,
            fill="none",
            stroke="currentColor",
            stroke_width="2",
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )


def _add_rect(
    dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float, rx: float
) -> None:
    dwg.add(
        dwg.rect(
            insert=(x, y),
            size=(w, h),
            rx=rx,
            fill="none",
            stroke="currentColor",
            stroke_width="2",
        )
    )


def _add_circle(dwg: svgwrite.Drawing, cx: float, cy: float, r: float) -> None:
    dwg.add(
        dwg.circle(
            center=(cx, cy),
            r=r,
            fill="none",
            stroke="currentColor",
            stroke_width="2",
        )
    )


def _build_file_text(dwg: svgwrite.Drawing) -> None:
    for d in (
        "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",
        "M14 2v4a2 2 0 0 0 2 2h4",
        "M10 9H8",
        "M16 13H8",
        "M16 17H8",
    ):
        _add_path(dwg, d)


def _build_sparkles(dwg: svgwrite.Drawing) -> None:
    for d in (
        "m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z",
        "M5 3v4",
        "M19 17v4",
        "M3 5h4",
        "M17 19h4",
    ):
        _add_path(dwg, d)


def _build_bot(dwg: svgwrite.Drawing) -> None:
    _add_path(dwg, "M12 8V4H8")
    _add_rect(dwg, 4, 8, 16, 12, 2)
    _add_path(dwg, "M2 14h2")
    _add_path(dwg, "M20 14h2")
    _add_path(dwg, "M15 13v2")
    _add_path(dwg, "M9 13v2")


def _build_user(dwg: svgwrite.Drawing) -> None:
    _add_circle(dwg, 12, 8, 5)
    _add_path(dwg, "M20 21a8 8 0 0 0-16 0")


def _build_folder(dwg: svgwrite.Drawing) -> None:
    _add_path(
        dwg,
        "M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z",
    )


def _build_pencil(dwg: svgwrite.Drawing) -> None:
    _add_path(
        dwg,
        "M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z",
    )
    _add_path(dwg, "m15 5 4 4")


def _build_trash(dwg: svgwrite.Drawing) -> None:
    for d in (
        "M3 6h18",
        "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",
        "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",
        "M10 11v6",
        "M14 11v6",
    ):
        _add_path(dwg, d)


def _build_download(dwg: svgwrite.Drawing) -> None:
    for d in (
        "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",
        "M7 10l5 5 5-5",
        "M12 15V3",
    ):
        _add_path(dwg, d)


def _build_plus(dwg: svgwrite.Drawing) -> None:
    _add_path(dwg, "M5 12h14")
    _add_path(dwg, "M12 5v14")


_BUILDERS: dict[str, Callable[[svgwrite.Drawing], None]] = {
    "file_text": _build_file_text,
    "sparkles": _build_sparkles,
    "bot": _build_bot,
    "user": _build_user,
    "folder": _build_folder,
    "pencil": _build_pencil,
    "trash": _build_trash,
    "download": _build_download,
    "plus": _build_plus,
}


def inline_icon_markup(name: str, *, size_em: str = "1.25em") -> str:
    """Return HTML fragment: span wrapping an SVG built with svgwrite."""
    builder = _BUILDERS.get(name)
    if builder is None:
        name = "folder"
        builder = _BUILDERS["folder"]
    # Lucide-style paths use minified tokens (e.g. `0-16`); svgwrite's debug validator rejects those.
    dwg = svgwrite.Drawing(size=(size_em, size_em), viewBox="0 0 24 24", debug=False)
    builder(dwg)
    raw = dwg.tostring()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return (
        f'<span style="display:inline-block;vertical-align:-0.2em;margin-right:0.25em">'
        f"{raw}</span>"
    )

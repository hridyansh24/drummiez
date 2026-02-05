#!/usr/bin/env python3
"""Create a simple PDF from the Markdown walkthrough without external PDF engines."""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

SOURCE = Path("Drummiez_AI_Explained.md")
TARGET = Path("Drummiez_AI_Explained.pdf")
PAGE_WIDTH = 612  # 8.5 inches * 72 pts
PAGE_HEIGHT = 792  # 11 inches * 72 pts
LEFT_MARGIN = 54   # 0.75 inch
TOP_MARGIN = 756   # start near top
FONT_SIZE = 10
LINE_HEIGHT = 14  # vertical distance per line
LINES_PER_PAGE = int((TOP_MARGIN - 36) / LINE_HEIGHT)

WRAP_WIDTH = 95


def load_wrapped_lines() -> list[str]:
    if not SOURCE.exists():
        raise SystemExit(f"Source file {SOURCE} does not exist")

    wrapper = textwrap.TextWrapper(
        width=WRAP_WIDTH,
        break_long_words=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )

    wrapped: list[str] = []
    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.rstrip("\n")
        if not stripped.strip():
            wrapped.append("")
            continue
        pieces = wrapper.wrap(stripped)
        if not pieces:
            wrapped.append("")
            continue
        wrapped.extend(pieces)
    return wrapped


def chunk_lines(lines: list[str]) -> list[list[str]]:
    if not lines:
        return [[]]
    return [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


def escape_pdf(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def build_page_stream(lines: list[str]) -> bytes:
    commands: list[str] = [
        "BT",
        f"/F1 {FONT_SIZE} Tf",
        f"{LINE_HEIGHT} TL",
        f"{LEFT_MARGIN} {TOP_MARGIN} Td",
    ]
    if not lines:
        commands.append("( ) Tj")
    for idx, line in enumerate(lines):
        safe = escape_pdf(line) if line else ""
        if idx == 0:
            commands.append(f"({safe or ' '}) Tj")
        else:
            commands.append("T*")
            commands.append(f"({safe or ' '}) Tj")
    commands.append("ET")
    stream_text = "\n".join(commands) + "\n"
    stream_bytes = stream_text.encode("utf-8")
    return stream_bytes


def build_pdf_objects(pages: list[list[str]]):
    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # Object 1: Catalog (points to Pages object #2)
    add_object(b"<< /Type /Catalog /Pages 2 0 R >>\n")

    # Placeholder for Pages object; fill later after kids list known
    objects.append(b"")  # will overwrite at the end

    # Object 3: Font definition (Courier)
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\n")

    # For each page, create content and page objects
    page_obj_ids = []
    content_obj_ids = []
    base_obj_index = 4
    for page_index, page_lines in enumerate(pages):
        stream_bytes = build_page_stream(page_lines)
        content_body = b"<< /Length %d >>\nstream\n" % len(stream_bytes)
        content_body += stream_bytes
        content_body += b"endstream\n"
        content_obj_id = add_object(content_body)
        content_obj_ids.append(content_obj_id)

        page_dict = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            b"/MediaBox [0 0 %d %d] " % (PAGE_WIDTH, PAGE_HEIGHT)
            + b"/Contents %d 0 R >>\n" % content_obj_id
        )
        page_obj_id = add_object(page_dict)
        page_obj_ids.append(page_obj_id)

    # Now fill the Pages object (object #2)
    kids_refs = b" ".join(f"{obj_id} 0 R".encode("ascii") for obj_id in page_obj_ids)
    pages_body = (
        b"<< /Type /Pages /Kids [ "
        + kids_refs
        + b" ] /Count %d >>\n" % len(page_obj_ids)
    )
    objects[1] = pages_body

    return objects


def write_pdf(objects: list[bytes]):
    with TARGET.open("wb") as handle:
        handle.write(b"%PDF-1.4\n")
        offsets = []
        for index, body in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{index} 0 obj\n".encode("ascii"))
            handle.write(body)
            handle.write(b"endobj\n")
        xref_start = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        handle.write(b"0000000000 65535 f \n")
        for offset in offsets:
            handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        handle.write(
            (
                "trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".format(
                    size=len(objects) + 1, start=xref_start
                ).encode("ascii")
            )
        )


def main():
    lines = load_wrapped_lines()
    pages = chunk_lines(lines)
    objects = build_pdf_objects(pages)
    write_pdf(objects)
    print(f"Wrote {TARGET} with {len(pages)} page(s)")


if __name__ == "__main__":
    main()

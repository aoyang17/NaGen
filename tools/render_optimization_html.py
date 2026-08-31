"""Render docs/OPTIMIZATION.md as a fully offline HTML table with SVG math."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
from pathlib import Path


MATH = re.compile(r"\\\((.+?)\\\)")


def parse_tables(lines: list[str]) -> list[list[list[str]]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("|"):
            current.append(line)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    tables = []
    for group in groups:
        rows = [
            [cell.strip() for cell in line.strip()[1:-1].split("|")]
            for line in group
        ]
        tables.append([rows[0], *rows[2:]])
    return tables


def inline_markup(text: str, rendered: iter[str]) -> str:
    pieces = []
    start = 0
    for match in MATH.finditer(text):
        pieces.append(html.escape(text[start:match.start()]))
        pieces.append(next(rendered))
        start = match.end()
    pieces.append(html.escape(text[start:]))
    value = "".join(pieces)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(
        r"\[\^([A-Za-z0-9_-]+)\]",
        r'<sup><a href="#fn-\1">1</a></sup>',
        value,
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mathjax-script", required=True)
    parser.add_argument("--node-modules", required=True)
    args = parser.parse_args()
    source = Path(args.input).read_text()
    lines = source.splitlines()
    title = next(line[2:] for line in lines if line.startswith("# "))
    formulas = MATH.findall(source)
    environment = dict(os.environ)
    environment["NODE_PATH"] = args.node_modules
    process = subprocess.run(
        ["node", args.mathjax_script],
        input=json.dumps(formulas),
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    rendered_values = json.loads(process.stdout)
    if len(rendered_values) != len(formulas):
        raise RuntimeError("MathJax output count does not match Markdown formulas")
    rendered = iter(rendered_values)
    table_html = []
    for table_index, table in enumerate(parse_tables(lines)):
        header, *rows = table
        body = []
        previous_category = None
        for row in rows:
            classes = []
            category = row[0]
            if table_index == 0 and category != previous_category:
                classes.append("major-group-start")
            if category == "优化目标" and previous_category is None:
                classes.append("objective-group-start")
            if row[1] == "结构弛豫":
                classes.append("post-generation-constraint")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            body.append(
                f"<tr{class_attr}>"
                + "".join(f"<td>{inline_markup(cell, rendered)}</td>" for cell in row)
                + "</tr>"
            )
            previous_category = category
        table_html.append(
            '<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )
    footnote_lines = [line for line in lines if line.startswith("[^")]
    footnotes = "".join(
        f'<li id="fn-{html.escape(line[2:line.index("]")])}">'
        f'{inline_markup(line.split(":", 1)[1].strip(), rendered)}</li>'
        for line in footnote_lines
    )
    try:
        next(rendered)
    except StopIteration:
        pass
    else:
        raise RuntimeError("unused MathJax renderings remain")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--line:#d9dfeb;--blue:#243d73;--page:#eef2f8}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--page);font-family:"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.55}}
main{{width:min(1500px,calc(100% - 40px));margin:28px auto 60px;padding:34px 38px 46px;background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 32px rgba(35,52,82,.09)}}
h1{{margin:0 0 24px;font-size:clamp(26px,3vw,40px);line-height:1.25}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin-top:18px}}
table{{width:100%;min-width:1100px;border-collapse:collapse}}th,td{{padding:12px 14px;text-align:left;vertical-align:top}}th{{color:#fff;background:var(--blue);font-weight:650;border-bottom:4px solid var(--ink)}}
td:first-child{{width:115px;font-weight:700;white-space:nowrap}}td:nth-child(2){{width:190px;font-weight:600}}td:nth-child(3){{min-width:430px}}tbody tr:nth-child(even){{background:#fafbfe}}tbody tr+tr td{{border-top:.6px solid #e8ecf3}}
tbody tr.major-group-start td{{border-top:4px solid var(--blue)}}tbody tr.objective-group-start td{{border-top:0}}tbody tr.post-generation-constraint td{{background:#fff8e6;border-top:3px solid #c47a00;border-bottom:3px solid #c47a00}}tbody tr.post-generation-constraint td:first-child{{border-left:3px solid #c47a00}}tbody tr.post-generation-constraint td:last-child{{border-right:3px solid #c47a00}}
mjx-container{{display:inline-block;max-width:100%;overflow-x:auto;overflow-y:hidden;vertical-align:middle}}mjx-container>svg{{max-width:none}}.footnotes{{margin-top:22px;padding-top:12px;border-top:1px solid var(--line);font-size:.9rem;color:#49566d}}
@media(max-width:900px){{main{{width:100%;margin:0;padding:22px 16px 36px;border:0;border-radius:0}}}}@media print{{body{{background:#fff}}main{{width:100%;margin:0;padding:0;border:0;box-shadow:none}}tr{{break-inside:avoid}}}}
</style></head><body><main><header><h1>{html.escape(title)}</h1></header>
{"".join(table_html)}<section class="footnotes"><ol>{footnotes}</ol></section>
</main></body></html>\n"""
    Path(args.output).write_text(document)
    print(json.dumps({
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "formula_count": len(formulas),
        "render_error_count": process.stderr.count("merror"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

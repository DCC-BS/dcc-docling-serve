# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab", "pillow", "fonttools", "numpy"]
# ///
"""Generate synthetic test PDFs with exact ground truth.

For every document three files are written to --out:
  <name>.pdf          the document
  <name>.gt.md        the markdown a perfect converter should produce (body content only)
  <name>.checks.json  text snippets tagged with where they live in the PDF:
                      body, heading, table, chart_text (vector chart, real text),
                      outlined_text (glyphs drawn as vector paths, no text layer),
                      bitmap_text (text inside an embedded raster image),
                      scan_text (full-page scan without text layer), furniture (header/footer)

    uv run --script benchmarks/tool_comparison/make_synthetic.py --out benchmarks/tool_comparison/synthetic
"""

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
from reportlab.pdfgen.canvas import Canvas

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
REGULAR, BOLD = FONT_DIR / "DejaVuSans.ttf", FONT_DIR / "DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(RLTTFont("Sans", str(REGULAR)))
pdfmetrics.registerFont(RLTTFont("Sans-Bold", str(BOLD)))
W, H = A4
MARGIN = 56


class Doc:
    """A PDF canvas plus its ground truth, filled side by side."""

    def __init__(self, name: str, out: Path):
        self.name, self.out = name, out
        self.c = Canvas(str(out / f"{name}.pdf"), pagesize=A4)
        self.c.setTitle(name)
        self.md: list[str] = []
        self.checks: list[dict] = []

    def check(self, category: str, text: str, page: int) -> None:
        self.checks.append({"category": category, "text": text, "page": page})

    def save(self) -> None:
        self.c.save()
        (self.out / f"{self.name}.gt.md").write_text("\n\n".join(self.md).strip() + "\n")
        (self.out / f"{self.name}.checks.json").write_text(json.dumps(self.checks, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- drawing helpers
def text(c, x, y, s, size=10.5, bold=False, color="#000000"):
    c.setFillColor(HexColor(color))
    c.setFont("Sans-Bold" if bold else "Sans", size)
    c.drawString(x, y, s)


def paragraph(c, x, y, s, width, size=10.5, leading=14, color="#000000") -> float:
    """Draw wrapped text, return the y below it."""
    for line in simpleSplit(s, "Sans", size, width):
        text(c, x, y, line, size, color=color)
        y -= leading
    return y - 6


class _PathPen(BasePen):
    def __init__(self, glyphset, path, x, y, scale):
        super().__init__(glyphset)
        self.path, self.x, self.y, self.scale = path, x, y, scale

    def _t(self, p):
        return self.x + p[0] * self.scale, self.y + p[1] * self.scale

    def _moveTo(self, p):
        self.path.moveTo(*self._t(p))

    def _lineTo(self, p):
        self.path.lineTo(*self._t(p))

    def _curveToOne(self, p1, p2, p3):
        self.path.curveTo(*self._t(p1), *self._t(p2), *self._t(p3))

    def _closePath(self):
        self.path.close()


_FONTS: dict[Path, TTFont] = {}


def outlined_text(c, x, y, s, size=12, bold=False, color="#000000"):
    """Draw text as vector glyph outlines: looks like text, has no text layer."""
    font = _FONTS.setdefault(BOLD if bold else REGULAR, TTFont(str(BOLD if bold else REGULAR)))
    glyphs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    scale = size / font["head"].unitsPerEm
    path = c.beginPath()
    for ch in s:
        name = cmap.get(ord(ch))
        if name is None:
            continue
        glyphs[name].draw(_PathPen(glyphs, path, x, y, scale))
        x += hmtx[name][0] * scale
    c.setFillColor(HexColor(color))
    c.drawPath(path, stroke=0, fill=1)


def table(
    c,
    x,
    y,
    rows,
    widths,
    row_h=18,
    size=9.5,
    grid=True,
    header_fill="#dde5ee",
    zebra=None,
    bold_header=True,
    header_color="#000000",
):
    """Draw a table; returns the y below it."""
    for r, row in enumerate(rows):
        top = y - r * row_h
        if r == 0 and header_fill:
            c.setFillColor(HexColor(header_fill))
            c.rect(x, top - row_h, sum(widths), row_h, stroke=0, fill=1)
        elif zebra and r % 2 == 0:
            c.setFillColor(HexColor(zebra))
            c.rect(x, top - row_h, sum(widths), row_h, stroke=0, fill=1)
        cx = x
        for cell, w in zip(row, widths, strict=True):
            if cell is not None:
                text(
                    c,
                    cx + 4,
                    top - row_h + 5,
                    cell,
                    size,
                    bold=(r == 0 and bold_header),
                    color=header_color if r == 0 else "#000000",
                )
            cx += w
    if grid:
        c.setStrokeColor(HexColor("#555555"))
        c.setLineWidth(0.6)
        for r in range(len(rows) + 1):
            c.line(x, y - r * row_h, x + sum(widths), y - r * row_h)
        cx = x
        for w in [0, *widths]:
            cx += w
            c.line(cx, y, cx, y - len(rows) * row_h)
    return y - len(rows) * row_h - 14


def md_table(rows) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(cell or "" for cell in row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def raster(img: Image.Image, fmt="PNG", quality=85) -> ImageReader:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, quality=quality)
    buffer.seek(0)
    return ImageReader(buffer)


# ---------------------------------------------------------------- document 1: charts
def doc_charts(out: Path) -> None:
    d = Doc("synthetic_charts", out)
    c = d.c
    title = "Quartalsbericht Energieverbrauch 2025"
    text(c, MARGIN, H - 80, title, 20, bold=True)
    d.md.append(f"# {title}")
    d.check("heading", title, 1)
    intro = (
        "Der Energieverbrauch der kantonalen Verwaltung ist im Jahr 2025 gegenüber dem Vorjahr um 4,2 Prozent "
        "gesunken. Die grössten Einsparungen wurden im dritten Quartal erzielt, als mehrere Gebäude an das "
        "Fernwärmenetz angeschlossen wurden."
    )
    y = paragraph(c, MARGIN, H - 115, intro, W - 2 * MARGIN)
    d.md.append(intro)
    d.check("body", intro, 1)

    heading = "Verbrauch nach Quartal"
    text(c, MARGIN, y - 10, heading, 14, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)

    # Vector bar chart with real-text labels
    chart_title = "Verbrauch pro Quartal (MWh)"
    x0, y0, cw, ch = MARGIN + 40, y - 290, 380, 220
    text(c, x0, y0 + ch + 20, chart_title, 11, bold=True)
    d.check("chart_text", chart_title, 1)
    c.setStrokeColor(HexColor("#333333"))
    c.line(x0, y0, x0 + cw, y0)
    c.line(x0, y0, x0, y0 + ch)
    values = {"Q1": 1240, "Q2": 980, "Q3": 760, "Q4": 1105}
    for i, tick in enumerate([0, 400, 800, 1200]):
        ty = y0 + tick / 1400 * ch
        c.setStrokeColor(HexColor("#cccccc"))
        c.line(x0, ty, x0 + cw, ty)
        text(c, x0 - 34, ty - 3, str(tick), 8)
        if i:
            d.check("chart_text", str(tick), 1)
    for i, (quarter, value) in enumerate(values.items()):
        bx = x0 + 30 + i * 88
        bh = value / 1400 * ch
        c.setFillColor(HexColor(["#2e86ab", "#f18f01", "#c73e1d", "#3b1f2b"][i]))
        c.rect(bx, y0, 50, bh, stroke=0, fill=1)
        text(c, bx + 14, y0 - 14, quarter, 9)
        text(c, bx + 8, y0 + bh + 5, f"{value:,}".replace(",", "'"), 8.5, bold=True)
        d.check("chart_text", quarter, 1)
        d.check("chart_text", f"{value:,}".replace(",", "'"), 1)
    caption = "Abbildung 1: Energieverbrauch der Verwaltung pro Quartal 2025."
    text(c, MARGIN, y0 - 40, caption, 9, color="#444444")
    d.md.append("<!-- image -->")
    d.md.append(caption)
    d.check("body", caption, 1)
    c.showPage()

    # Page 2: line chart whose labels are outlines, pie chart with real-text legend
    heading = "Temperatur und Energiemix"
    text(c, MARGIN, H - 80, heading, 14, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 2)
    x0, y0, cw, ch = MARGIN + 40, H - 350, 400, 200
    outlined_text(c, x0, y0 + ch + 18, "Temperaturverlauf Basel 2025", 11, bold=True)
    d.check("outlined_text", "Temperaturverlauf Basel 2025", 2)
    c.setStrokeColor(HexColor("#333333"))
    c.line(x0, y0, x0 + cw, y0)
    c.line(x0, y0, x0, y0 + ch)
    temps = [2.1, 4.8, 9.6, 13.9, 18.2, 22.7]
    months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun"]
    for tick in (0, 10, 20):
        ty = y0 + tick / 25 * ch
        outlined_text(c, x0 - 34, ty - 3, f"{tick} °C", 8)
        d.check("outlined_text", f"{tick} °C", 2)
    path = c.beginPath()
    for i, (month, temp) in enumerate(zip(months, temps, strict=True)):
        px, py = x0 + 30 + i * 68, y0 + temp / 25 * ch
        path.moveTo(px, py) if i == 0 else path.lineTo(px, py)
        outlined_text(c, px - 10, y0 - 14, month, 9)
        d.check("outlined_text", month, 2)
    c.setStrokeColor(HexColor("#c73e1d"))
    c.setLineWidth(2)
    c.drawPath(path, stroke=1, fill=0)
    c.setLineWidth(1)
    caption = "Abbildung 2: Monatsmittel der Lufttemperatur (Beschriftung als Vektorpfade)."
    text(c, MARGIN, y0 - 38, caption, 9, color="#444444")
    d.md.append("<!-- image -->")
    d.md.append(caption)
    d.check("body", caption, 2)

    # Pie chart with legend as real text
    cx_, cy_, r = MARGIN + 110, y0 - 170, 70
    shares = [("Solar", 42, "#f18f01"), ("Wind", 31, "#2e86ab"), ("Wasserkraft", 27, "#3b8e5a")]
    start = 90
    for _label, share, color in shares:
        extent = share / 100 * 360
        c.setFillColor(HexColor(color))
        c.wedge(cx_ - r, cy_ - r, cx_ + r, cy_ + r, start, extent, stroke=0, fill=1)
        start += extent
    for i, (label, share, color) in enumerate(shares):
        ly = cy_ + 30 - i * 22
        c.setFillColor(HexColor(color))
        c.rect(cx_ + 100, ly - 2, 12, 12, stroke=0, fill=1)
        text(c, cx_ + 120, ly, f"{label} {share} %", 10)
        d.check("chart_text", f"{label} {share} %", 2)
    closing = (
        "Der Anteil erneuerbarer Energien am zugekauften Strom lag bei 100 Prozent. Für 2026 ist ein weiterer "
        "Ausbau der Photovoltaik auf Schulhausdächern geplant."
    )
    paragraph(c, MARGIN, cy_ - 110, closing, W - 2 * MARGIN)
    d.md.append("<!-- image -->")
    d.md.append(closing)
    d.check("body", closing, 2)
    d.save()


# ---------------------------------------------------------------- document 2: tables and forms
def doc_tables(out: Path) -> None:
    d = Doc("synthetic_tables", out)
    c = d.c

    def furniture(page: int):
        header, footer = "Statistisches Amt – Tabellenbericht", f"Seite {page} von 2"
        text(c, MARGIN, H - 36, header, 8, color="#666666")
        text(c, W - MARGIN - 50, 30, footer, 8, color="#666666")
        c.setStrokeColor(HexColor("#999999"))
        c.line(MARGIN, H - 42, W - MARGIN, H - 42)
        d.check("furniture", header, page)
        d.check("furniture", footer, page)

    furniture(1)
    title = "Bevölkerung und Wohnen in Basel-Stadt"
    text(c, MARGIN, H - 80, title, 18, bold=True)
    d.md.append(f"# {title}")
    d.check("heading", title, 1)

    heading = "Tabelle mit Rahmen"
    text(c, MARGIN, H - 115, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)
    rows = [
        ["Wohnviertel", "Einwohner", "Fläche km²", "Dichte"],
        ["Altstadt Grossbasel", "2'134", "0.37", "5'768"],
        ["Vorstädte", "4'512", "0.99", "4'558"],
        ["Am Ring", "10'873", "1.07", "10'162"],
        ["Breite", "8'921", "0.86", "10'373"],
        ["St. Johann", "20'317", "2.51", "8'094"],
    ]
    y = table(c, MARGIN, H - 128, rows, [160, 90, 90, 90])
    d.md.append(md_table(rows))
    for row in rows:
        d.check("table", " ".join(row), 1)

    heading = "Tabelle ohne Rahmen"
    text(c, MARGIN, y - 6, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)
    rows2 = [
        ["Jahr", "Geburten", "Todesfälle", "Saldo"],
        ["2022", "1'861", "1'912", "-51"],
        ["2023", "1'790", "1'874", "-84"],
        ["2024", "1'822", "1'801", "21"],
    ]
    y = table(c, MARGIN, y - 20, rows2, [100, 100, 100, 100], grid=False, header_fill=None)
    d.md.append(md_table(rows2))
    for row in rows2:
        d.check("table", " ".join(row), 1)

    heading = "Tabelle mit verbundenen Kopfzellen"
    text(c, MARGIN, y - 6, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)
    ty = y - 20
    widths = [140, 85, 85, 85, 85]
    # Two-level header: "Schweiz" spans 2 columns, "Ausland" spans 2 columns
    c.setFillColor(HexColor("#dde5ee"))
    c.rect(MARGIN, ty - 36, sum(widths), 36, stroke=0, fill=1)
    text(c, MARGIN + 4, ty - 24, "Altersgruppe", 9.5, bold=True)
    text(c, MARGIN + 140 + 60, ty - 13, "Schweiz", 9.5, bold=True)
    text(c, MARGIN + 310 + 60, ty - 13, "Ausland", 9.5, bold=True)
    for i, label in enumerate(["Männer", "Frauen", "Männer", "Frauen"]):
        text(c, MARGIN + 144 + i * 85, ty - 31, label, 9.5, bold=True)
    body = [["0–19 Jahre", "9'812", "9'344", "4'120", "3'987"], ["20–64 Jahre", "38'201", "37'655", "21'433", "19'870"]]
    y = table(c, MARGIN, ty - 36, body, widths, header_fill=None, bold_header=False)
    # Header grid: outer box, spanning cells on top, sub-columns below
    c.setStrokeColor(HexColor("#555555"))
    c.setLineWidth(0.6)
    c.rect(MARGIN, ty - 36, sum(widths), 36, stroke=1, fill=0)
    c.line(MARGIN + 140, ty - 18, MARGIN + sum(widths), ty - 18)
    for cx in (140, 310):
        c.line(MARGIN + cx, ty, MARGIN + cx, ty - 36)
    for cx in (225, 395):
        c.line(MARGIN + cx, ty - 18, MARGIN + cx, ty - 36)
    merged_md = [
        ["Altersgruppe", "Schweiz Männer", "Schweiz Frauen", "Ausland Männer", "Ausland Frauen"],
        *body,
    ]
    d.md.append(md_table(merged_md))
    for row in merged_md:
        d.check("table", " ".join(row), 1)

    note = "¹ Quelle: Kantonales Einwohnerregister, Stichtag 31. Dezember 2024."
    text(c, MARGIN, y - 10, note, 8, color="#444444")
    d.md.append(note)
    d.check("body", note, 1)
    c.showPage()

    # Page 2: form, two columns, lists
    furniture(2)
    heading = "Antragsformular"
    text(c, MARGIN, H - 80, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 2)
    fields = [("Name", "Muster"), ("Vorname", "Anna"), ("Geburtsdatum", "12.03.1985"), ("Gemeinde", "Riehen")]
    fy = H - 110
    form_md = []
    for label, value in fields:
        text(c, MARGIN, fy, f"{label}:", 10, bold=True)
        c.setStrokeColor(HexColor("#555555"))
        c.rect(MARGIN + 110, fy - 5, 200, 18, stroke=1, fill=0)
        text(c, MARGIN + 116, fy, value, 10)
        form_md.append(f"{label}: {value}")
        d.check("body", f"{label} {value}", 2)
        fy -= 26
    c.rect(MARGIN, fy - 2, 10, 10, stroke=1, fill=0)
    c.line(MARGIN + 2, fy + 3, MARGIN + 5, fy)
    c.line(MARGIN + 5, fy, MARGIN + 9, fy + 7)
    text(c, MARGIN + 16, fy, "Ich bestätige die Richtigkeit der Angaben.", 10)
    form_md.append("- [x] Ich bestätige die Richtigkeit der Angaben.")
    d.check("body", "Ich bestätige die Richtigkeit der Angaben.", 2)
    d.md.append("\n\n".join(form_md))

    heading = "Zweispaltiger Text"
    text(c, MARGIN, fy - 36, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 2)
    col_w = (W - 2 * MARGIN - 20) / 2
    left = (
        "Die linke Spalte beschreibt die Methodik. Alle Angaben beruhen auf dem kantonalen Einwohnerregister "
        "und werden jährlich per Stichtag ausgewertet. Personen im Asylprozess sind nicht enthalten."
    )
    right = (
        "Die rechte Spalte fasst die Resultate zusammen. Die Bevölkerung wuchs um 0,8 Prozent, getrieben durch "
        "Zuwanderung. Die Geburtenzahl blieb stabil, die Zahl der Todesfälle sank leicht."
    )
    paragraph(c, MARGIN, fy - 58, left, col_w)
    ly = paragraph(c, MARGIN + col_w + 20, fy - 58, right, col_w)
    d.md += [left, right]
    d.check("body", left, 2)
    d.check("body", right, 2)

    heading = "Aufzählungen"
    text(c, MARGIN, ly - 40, heading, 13, bold=True)
    d.md.append(f"## {heading}")
    d.check("heading", heading, 2)
    bullets = ["Wohnungsbestand nach Zimmerzahl", "Leerstandsquote nach Wohnviertel", "Mietpreise pro Quadratmeter"]
    numbered = ["Daten erheben", "Plausibilisieren", "Publizieren"]
    by = ly - 62
    for item in bullets:
        text(c, MARGIN + 6, by, "•", 10)
        text(c, MARGIN + 20, by, item, 10)
        d.check("body", item, 2)
        by -= 16
    by -= 8
    for i, item in enumerate(numbered, 1):
        text(c, MARGIN + 4, by, f"{i}.", 10)
        text(c, MARGIN + 20, by, item, 10)
        d.check("body", item, 2)
        by -= 16
    d.md.append("\n".join(f"- {b}" for b in bullets))
    d.md.append("\n".join(f"{i}. {n}" for i, n in enumerate(numbered, 1)))
    formula = "Dichte = Einwohner / Fläche"
    text(c, MARGIN + 150, by - 20, formula, 11)
    d.md.append(formula)
    d.check("body", formula, 2)
    d.save()


# ---------------------------------------------------------------- document 3: scan and embedded bitmaps
SCAN_LETTER = [
    "Basel, 14. Februar 2025",
    "Betreff: Gesuch um Verlängerung der Bewilligung",
    "Sehr geehrte Damen und Herren",
    "Hiermit beantrage ich die Verlängerung der Bewilligung für den Marktstand auf dem Barfüsserplatz "
    "um weitere zwölf Monate. Der Stand wird unverändert an Samstagen von 8 bis 16 Uhr betrieben.",
    "Die geforderten Unterlagen, namentlich die Haftpflichtversicherung und die Bestätigung der "
    "Lebensmittelkontrolle, liegen diesem Schreiben bei.",
    "Freundliche Grüsse",
    "Peter Beispiel",
]


def doc_scan(out: Path) -> None:
    d = Doc("synthetic_scan", out)
    c = d.c
    rng = random.Random(7)
    # Page 1: a scanned letter, no text layer
    dpi = 200
    img = Image.new("L", (int(W / 72 * dpi), int(H / 72 * dpi)), 250)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(REGULAR), 34)
    bold = ImageFont.truetype(str(BOLD), 34)
    y = 260
    for i, line in enumerate(SCAN_LETTER):
        f = bold if line.startswith("Betreff") else font
        wrapped = []
        words, current = line.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=f) > img.width - 400:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
        for part in wrapped:
            draw.text((200, y), part, fill=25, font=f)
            y += 48
        y += 40 if i in (0, 1, 2, 4, 5) else 20
    img = img.rotate(0.6, expand=False, fillcolor=250)
    noise = np.random.default_rng(3).normal(0, 9, (img.height, img.width))
    img = Image.fromarray(np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype("uint8"))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    c.drawImage(raster(img, "JPEG", 70), 0, 0, W, H)
    d.md += [SCAN_LETTER[0], f"**{SCAN_LETTER[1]}**", *SCAN_LETTER[2:]]
    for line in SCAN_LETTER:
        d.check("scan_text", line, 1)
    c.showPage()

    # Page 2: digital text + screenshot bitmap with text + photo without text
    heading = "Betriebsbericht Informatik"
    text(c, MARGIN, H - 80, heading, 16, bold=True)
    d.md.append(f"# {heading}")
    d.check("heading", heading, 2)
    body = (
        "Die folgende Abbildung zeigt einen Ausschnitt aus dem Überwachungssystem. Alle Dienste waren im "
        "Berichtsmonat verfügbar, mit Ausnahme einer geplanten Wartung am 9. März."
    )
    y = paragraph(c, MARGIN, H - 110, body, W - 2 * MARGIN)
    d.md.append(body)
    d.check("body", body, 2)

    shot = Image.new("RGB", (900, 330), "#f4f6f8")
    sd = ImageDraw.Draw(shot)
    sd.rectangle([0, 0, 900, 50], fill="#1f3b57")
    sf, sb = ImageFont.truetype(str(REGULAR), 24), ImageFont.truetype(str(BOLD), 26)
    sd.text((20, 10), "Dienststatus Rechenzentrum", fill="white", font=sb)
    d.check("bitmap_text", "Dienststatus Rechenzentrum", 2)
    services = [
        ("Webportal", "Verfügbar", "99.98 %"),
        ("E-Mail", "Verfügbar", "100.00 %"),
        ("Archiv", "Wartung", "97.40 %"),
    ]
    for i, (svc, state, uptime) in enumerate(services):
        yy = 80 + i * 75
        sd.rectangle([20, yy, 880, yy + 60], outline="#c8d0d8", fill="white")
        sd.ellipse([35, yy + 18, 59, yy + 42], fill="#2e9d57" if state == "Verfügbar" else "#e0a100")
        sd.text((80, yy + 16), svc, fill="#222", font=sf)
        sd.text((420, yy + 16), state, fill="#222", font=sf)
        sd.text((700, yy + 16), uptime, fill="#222", font=sf)
        d.check("bitmap_text", f"{svc} {state} {uptime}", 2)
    iw = W - 2 * MARGIN
    ih = iw * shot.height / shot.width
    c.drawImage(raster(shot), MARGIN, y - ih - 10, iw, ih)
    caption = "Abbildung 3: Bildschirmfoto des Überwachungssystems."
    text(c, MARGIN, y - ih - 28, caption, 9, color="#444444")
    d.md += ["<!-- image -->", caption]
    d.check("body", caption, 2)

    photo = Image.new("RGB", (600, 360))
    pd_ = ImageDraw.Draw(photo)
    for yy in range(360):
        pd_.line([(0, yy), (600, yy)], fill=(40 + yy // 4, 90 + yy // 5, 160 - yy // 6))
    for _ in range(40):
        x, yy, r = rng.randint(0, 600), rng.randint(0, 360), rng.randint(5, 40)
        pd_.ellipse([x - r, yy - r, x + r, yy + r], fill=(rng.randint(120, 255), rng.randint(120, 255), 90))
    py = y - ih - 60
    c.drawImage(raster(photo, "JPEG"), MARGIN, py - 200, 330, 198)
    caption2 = "Abbildung 4: Serverraum (Symbolbild)."
    text(c, MARGIN, py - 216, caption2, 9, color="#444444")
    d.md += ["<!-- image -->", caption2]
    d.check("body", caption2, 2)
    d.save()


# ---------------------------------------------------------------- document 4: designed page
def doc_design(out: Path) -> None:
    d = Doc("synthetic_design", out)
    c = d.c
    # Full-page background fill (the case that sends whole pages to OCR in docling)
    c.setFillColor(HexColor("#f7ede2"))
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(HexColor("#1f3b57"))
    c.rect(0, H - 70, W, 70, stroke=0, fill=1)
    text(c, MARGIN, H - 42, "Kanton Basel-Stadt  |  Präsidialdepartement", 11, color="#ffffff")
    d.check("furniture", "Kanton Basel-Stadt Präsidialdepartement", 1)

    # Vector logo, no text
    c.setFillColor(HexColor("#c73e1d"))
    c.circle(W - MARGIN - 20, H - 35, 18, stroke=0, fill=1)
    c.setFillColor(white)
    c.circle(W - MARGIN - 20, H - 35, 8, stroke=0, fill=1)

    title = "Jahresrückblick 2025"
    outlined_text(c, MARGIN, H - 130, title, 30, bold=True, color="#1f3b57")
    d.md.append(f"# {title}")
    d.check("outlined_text", title, 1)
    d.check("heading", title, 1)
    subtitle = "Die wichtigsten Kennzahlen auf einen Blick"
    text(c, MARGIN, H - 155, subtitle, 13, color="#444444")
    d.md.append(subtitle)
    d.check("body", subtitle, 1)

    # KPI boxes: white text on coloured fills
    kpis = [("201'971", "Einwohnerinnen und Einwohner"), ("37,1 km²", "Kantonsfläche"), ("1'482", "Neue Wohnungen")]
    bw = (W - 2 * MARGIN - 20) / 3
    kpi_md = []
    for i, (value, label) in enumerate(kpis):
        bx = MARGIN + i * (bw + 10)
        c.setFillColor(HexColor(["#2e86ab", "#3b8e5a", "#c73e1d"][i]))
        c.roundRect(bx, H - 270, bw, 90, 8, stroke=0, fill=1)
        text(c, bx + 12, H - 215, value, 20, bold=True, color="#ffffff")
        for j, line in enumerate(simpleSplit(label, "Sans", 9.5, bw - 24)):
            text(c, bx + 12, H - 238 - j * 12, line, 9.5, color="#ffffff")
        kpi_md.append(f"- **{value}** {label}")
        d.check("body", f"{value} {label}", 1)
    d.md.append("\n".join(kpi_md))

    heading = "Schwerpunkte des Jahres"
    text(c, MARGIN, H - 305, heading, 14, bold=True, color="#1f3b57")
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)
    # Text inside a light box with a rule on the left (shapes around real text)
    box_text = (
        "Im Zentrum standen die Umsetzung der Klimastrategie, der Ausbau der Tagesstrukturen an den "
        "Primarschulen und die Digitalisierung der Baugesuche. Alle drei Vorhaben wurden termingerecht "
        "abgeschlossen oder befinden sich in der planmässigen Umsetzung."
    )
    c.setFillColor(HexColor("#ffffff"))
    c.rect(MARGIN, H - 400, W - 2 * MARGIN, 80, stroke=0, fill=1)
    c.setFillColor(HexColor("#f18f01"))
    c.rect(MARGIN, H - 400, 5, 80, stroke=0, fill=1)
    paragraph(c, MARGIN + 16, H - 338, box_text, W - 2 * MARGIN - 30)
    d.md.append(box_text)
    d.check("body", box_text, 1)

    # Zebra table without vertical rules on the coloured background
    heading = "Budget nach Departement"
    text(c, MARGIN, H - 430, heading, 14, bold=True, color="#1f3b57")
    d.md.append(f"## {heading}")
    d.check("heading", heading, 1)
    rows = [
        ["Departement", "Budget Mio. CHF", "Veränderung"],
        ["Bau- und Verkehrsdepartement", "412.5", "+2.1 %"],
        ["Erziehungsdepartement", "1'104.8", "+3.4 %"],
        ["Gesundheitsdepartement", "688.0", "-0.7 %"],
        ["Präsidialdepartement", "96.3", "+1.2 %"],
    ]
    y = table(c, MARGIN, H - 442, rows, [220, 130, 130], grid=False, header_fill="#1f3b57", zebra="#efe1d1",
              header_color="#ffffff")  # fmt: skip
    d.md.append(md_table(rows))
    for row in rows:
        d.check("table", " ".join(row), 1)

    # Icon row: vector icons with real-text labels
    icons = ["Mobilität", "Bildung", "Gesundheit", "Kultur"]
    for i, label in enumerate(icons):
        ix = MARGIN + 20 + i * 120
        c.setFillColor(HexColor("#1f3b57"))
        c.circle(ix + 15, y - 40, 15, stroke=0, fill=1)
        c.setFillColor(HexColor("#f7ede2"))
        for k in range(5):
            angle = 2 * math.pi * k / 5
            c.circle(ix + 15 + 8 * math.cos(angle), y - 40 + 8 * math.sin(angle), 2, stroke=0, fill=1)
        text(c, ix, y - 72, label, 10)
        d.check("body", label, 1)
    d.md.append(" ".join(icons))

    footer = "statistik.bs.ch  ·  Seite 1"
    c.setFillColor(Color(0.12, 0.23, 0.34))
    c.rect(0, 0, W, 34, stroke=0, fill=1)
    text(c, MARGIN, 13, footer, 8.5, color="#ffffff")
    d.check("furniture", "statistik.bs.ch Seite 1", 1)
    d.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for build in (doc_charts, doc_tables, doc_scan, doc_design):
        build(args.out)
        print("wrote", build.__name__)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab", "pillow", "numpy", "pypdfium2"]
# ///
"""Generate list test PDFs with exact ground truth.

Every document gets <name>.pdf and <name>.gt.json:
  items  the list items a perfect converter returns as list items (marker stripped)
  traps  lines that start like a list item but are not one (headings, dates, years, footnotes)

The *_scan variants are the same pages rendered to a noisy, slightly rotated image without a
text layer, so list detection has to work on OCR text.

    uv run --script benchmarks/list_eval/make_lists.py --out benchmarks/list_eval/docs
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
W, H = A4
MARGIN = 60


class Doc:
    def __init__(self, name: str, out: Path):
        self.name, self.out = name, out
        self.c = Canvas(str(out / f"{name}.pdf"), pagesize=A4)
        self.c.setTitle(name)
        self.items: list[dict] = []
        self.traps: list[str] = []
        self.page = 1

    def new_page(self) -> float:
        self.c.showPage()
        self.page += 1
        return H - MARGIN

    def save(self) -> None:
        self.c.save()
        gt = {"items": self.items, "traps": self.traps}
        (self.out / f"{self.name}.gt.json").write_text(
            json.dumps(gt, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )


def lines(c, x, y, s, width, size=10.5, leading=14, bold=False) -> float:
    font = "Sans-Bold" if bold else "Sans"
    c.setFont(font, size)
    for line in simpleSplit(s, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def heading(d: Doc, x, y, s, size=13) -> float:
    d.traps.append(s)
    return lines(d.c, x, y, s, W, size=size, leading=size + 8, bold=True) - 4


def para(d: Doc, x, y, s, width, trap=False) -> float:
    if trap:
        d.traps.append(s)
    return lines(d.c, x, y, s, width) - 8


def items(d: Doc, x, y, marker, texts, width, indent=16, gap=3) -> float:
    """Draw a list; marker is a glyph or a callable(i) returning the enumerator."""
    for i, t in enumerate(texts):
        m = marker(i) if callable(marker) else marker
        d.c.setFont("Sans", 10.5)
        d.c.drawString(x, y, m)
        y = lines(d.c, x + indent, y, t, width - indent) - gap
        d.items.append({"text": t, "page": d.page, "marker": m})
    return y - 8


# ---------------------------------------------------------------- document 1: marker styles and traps
def doc_styles(out: Path) -> None:
    d = Doc("lists_styles", out)
    c, x, w = d.c, MARGIN, W - 2 * MARGIN
    y = H - MARGIN
    y = heading(d, x, y, "1. Einleitung", 15)
    y = para(d, x, y, "Der Regierungsrat hat im Berichtsjahr mehrere Massnahmen beschlossen. Die wichtigsten "
             "Punkte sind in den folgenden Listen zusammengefasst.", w)  # fmt: skip
    y = items(d, x, y, "•", [
        "Ausbau der Tagesbetreuung in allen Quartieren bis Ende 2027",
        "Senkung der Energiekosten in den kantonalen Liegenschaften um mindestens zwölf Prozent gegenüber "
        "dem Durchschnitt der Jahre 2019 bis 2023, gemessen am witterungsbereinigten Verbrauch",
        "Digitalisierung der Baubewilligungsverfahren",
    ], w)  # fmt: skip
    y = para(d, x, y, "2025 war ein Jahr mit vielen Veränderungen in der Verwaltung, die sich auch im "
             "Budget niedergeschlagen haben.", w, trap=True)  # fmt: skip
    y = heading(d, x, y, "2. Vorgehen")
    y = items(d, x, y, lambda i: f"{i + 1}.", [
        "Bestandesaufnahme der bestehenden Angebote",
        "Befragung der Anspruchsgruppen in allen drei Gemeinden",
        "Auswertung und Bericht an den Grossen Rat mit Empfehlungen für die nächste Legislatur",
        "Umsetzung der beschlossenen Massnahmen",
    ], w)  # fmt: skip
    y = para(d, x, y, "12. März 2025: Der Grosse Rat hat den Bericht zur Kenntnis genommen und die "
             "Regierung beauftragt, bis Ende Jahr einen Umsetzungsplan vorzulegen.", w, trap=True)  # fmt: skip
    y = heading(d, x, y, "2.1 Anforderungen an die Daten")
    y = items(d, x, y, lambda i: f"{'abcde'[i]})", [
        "Aggregatzustand und Farbe",
        "Schmelzpunkt oder Gefrierpunkt",
        "Siedebeginn und Siedebereich, sofern der Stoff bei Normaldruck siedet",
    ], w)  # fmt: skip
    y = items(d, x + 16, y, "–", [
        "Angaben in Grad Celsius",
        "Messmethode nach Anhang VII",
    ], w - 16)  # fmt: skip
    y = para(d, x, y, "Die Tabelle unten zeigt die Veränderung gegenüber dem Vorjahr.", w)
    rows = [("Position", "2024", "Veränderung"), ("Personal", "12'400", "- 1'200"), ("Sachaufwand", "8'150", "+ 300")]
    for r in rows:
        for j, cell in enumerate(r):
            c.setFont("Sans-Bold" if r is rows[0] else "Sans", 10)
            c.drawString(x + j * 150, y, cell)
        y -= 16
    d.traps.append("- 1'200")
    y -= 10
    c.setFont("Sans", 8)
    c.drawString(x, 70, "1 Quelle: Statistisches Amt Basel-Stadt, Erhebung 2025")
    d.traps.append("1 Quelle: Statistisches Amt Basel-Stadt, Erhebung 2025")

    y = d.new_page()
    y = heading(d, x, y, "3. Weitere Formen")
    y = items(d, x, y, lambda i: f"({['i', 'ii', 'iii'][i]})", [
        "die Hersteller und Importeure registrierter Stoffe",
        "die nachgeschalteten Anwender, die den Stoff in einem Gemisch verwenden",
        "die Händler",
    ], w, indent=26)  # fmt: skip
    y = items(d, x, y, lambda i: f"{i + 1})", [
        "Antrag einreichen",
        "Unterlagen nachreichen, falls die Behörde dies innert 30 Tagen verlangt",
    ], w)  # fmt: skip
    y = items(d, x, y, "–", [
        "Kinder und Jugendliche",
        "Seniorinnen und Senioren",
        "Menschen mit Behinderungen",
    ], w)  # fmt: skip
    y = items(d, x, y, "■", ["Sitzung vom 4. Juni", "Sitzung vom 2. Juli"], w)
    y = items(d, x, y, "✓", ["Budget genehmigt", "Stellenplan bereinigt"], w)
    y = para(d, x, y, "3 Varianten wurden geprüft; die Kommission empfiehlt die zweite.", w, trap=True)
    y = para(d, x, y, "Zusammenfassend lässt sich festhalten, dass die Massnahmen wirken.", w)
    d.save()


# ---------------------------------------------------------------- document 2: two columns, dense lists
def doc_columns(out: Path) -> None:
    d = Doc("lists_columns", out)
    c = d.c
    colw = (W - 2 * MARGIN - 24) / 2
    y = heading(d, MARGIN, H - MARGIN, "Angebote im Überblick", 15)
    top = y
    for col, (title, marker, texts) in enumerate([
        ("Beratung", "•", [
            "Budgetberatung für Familien mit kleinem Einkommen",
            "Schuldenberatung, auch anonym und telefonisch",
            "Rechtsauskunft zu Miete und Arbeit",
            "Begleitung bei Behördengängen",
            "Vermittlung von Dolmetscherinnen und Dolmetschern für über zwanzig Sprachen",
        ]),
        ("Kurse", "–", [
            "Deutsch im Alltag, Stufen A1 bis B1",
            "Computer-Grundkurs",
            "Bewerbungstraining mit Einzelcoaching",
            "Elternbildung zu Medien und Schule",
        ]),
    ]):  # fmt: skip
        x = MARGIN + col * (colw + 24)
        yy = heading(d, x, top, title, 12)
        yy = para(d, x, yy, "Die Angebote sind kostenlos und stehen allen Einwohnerinnen und Einwohnern offen.", colw)
        yy = items(d, x, yy, marker, texts, colw, gap=1)
        yy = para(d, x, yy, "Anmeldung über das Quartierbüro.", colw)
    y = 380
    y = heading(d, MARGIN, y, "Termine", 12)
    y = items(d, MARGIN, y, lambda i: f"{i + 1}.", [
        "Informationsabend am 3. September",
        "Anmeldeschluss am 20. September",
        "Kursbeginn am 7. Oktober",
    ], W - 2 * MARGIN, gap=0)  # fmt: skip
    y = para(d, MARGIN, y, "Für Fragen steht das Sekretariat von Montag bis Freitag zur Verfügung.", W - 2 * MARGIN)
    d.save()


def scan(out: Path, name: str) -> None:
    """Render a document to a noisy, slightly rotated image-only PDF."""
    src = pdfium.PdfDocument(out / f"{name}.pdf")
    c = Canvas(str(out / f"{name}_scan.pdf"), pagesize=A4)
    rng = np.random.default_rng(7)
    for i in range(len(src)):
        img = src[i].render(scale=200 / 72).to_pil().convert("L")
        img = img.rotate(0.5 if i % 2 == 0 else -0.4, expand=False, fillcolor=250)
        noise = rng.normal(0, 8, (img.height, img.width))
        img = Image.fromarray(np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, W, H)
        c.showPage()
    c.save()
    gt = json.loads((out / f"{name}.gt.json").read_text())
    (out / f"{name}_scan.gt.json").write_text(json.dumps(gt, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    out = p.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    doc_styles(out)
    doc_columns(out)
    for name in ("lists_styles", "lists_columns"):
        scan(out, name)


if __name__ == "__main__":
    main()

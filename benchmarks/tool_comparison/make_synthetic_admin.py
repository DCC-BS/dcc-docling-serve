# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab", "pillow", "fonttools", "numpy", "pypdfium2", "segno"]
# ///
"""Generate synthetic administrative documents with exact ground truth, clean and degraded.

Typical paperwork that reaches an administration: a salary certificate, an electricity bill
and a craftsman's invoice with a QR payment slip, a till receipt, an official decision letter,
a handwritten application form and a health-insurance statement. All names, numbers and
companies are made up; every page says so in its footer.

Each document is written as a born-digital PDF (where one exists in reality) and as degraded
copies rendered from it, so the ground truth stays exact:
  _scan    200 dpi office scan: slight rotation, noise, blur, JPEG
  _fax     fax quality: half vertical resolution, binarised, speckles
  _thermal faded till receipt on thermal paper
  _photo   phone photo (JPG): perspective, uneven light, background

Output per document, next to make_synthetic.py's files in --out:
  <name>.pdf|.jpg, <name>.gt.md, <name>.checks.json (categories as in make_synthetic.py, plus
  handwriting and photo_text)

    uv run --script benchmarks/tool_comparison/make_synthetic_admin.py
"""

import argparse
import io
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pypdfium2
import segno
from make_synthetic import BOLD, MARGIN, Doc, md_table, paragraph, raster, table, text
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
from reportlab.pdfgen.canvas import Canvas

W, H = A4
MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
pdfmetrics.registerFont(RLTTFont("Mono", str(MONO)))
FOOTER = "Testdokument mit fiktiven Daten"

# Handwriting fonts (SIL Open Font License) from google/fonts at pinned commits, cached locally
FONT_CACHE = Path.home() / ".cache" / "dcc-benchmarks" / "fonts"
HAND_FONTS = {
    "Caveat.ttf": "https://raw.githubusercontent.com/google/fonts/5571d84c0d8c70ec1af4f64072d8c5cf1e4e9643/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "Kalam.ttf": "https://raw.githubusercontent.com/google/fonts/6e4b84c976cadb3c49a40fd9a1c203e4f7fcf2da/ofl/kalam/Kalam-Regular.ttf",
}


def hand_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_CACHE / name
    if not path.exists():
        FONT_CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HAND_FONTS[name], timeout=60) as response:  # noqa: S310 -- pinned https URL
            path.write_bytes(response.read())
    return ImageFont.truetype(str(path), size)


def handwriting(s: str, font: str = "Caveat.ttf", size: int = 64, color=(20, 40, 120)) -> Image.Image:
    """Text in a handwriting font as a transparent bitmap, slightly tilted."""
    f = hand_font(font, size)
    box = f.getbbox(s)
    img = Image.new("RGBA", (box[2] + 20, box[3] + 24), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((10, 6), s, font=f, fill=(*color, 255))
    return img.rotate(1.2, expand=True, resample=Image.BICUBIC)


def draw_hand(d: Doc, x, y, s, page, height=16, font="Caveat.ttf", category="handwriting"):
    img = handwriting(s, font)
    width = img.width * height / img.height
    d.c.drawImage(raster(img), x, y, width, height, mask="auto")
    d.check(category, s, page)


def footer(d: Doc, page: int, width=W) -> None:
    text(d.c, MARGIN if width > 300 else 8, 20, f"{FOOTER} – Seite {page}", 7, color="#777777")
    d.check("furniture", FOOTER, page)


def heading(d: Doc, s, page, x=MARGIN, y=None, size=15, level=1):
    text(d.c, x, y, s, size, bold=True)
    d.md.append(f"{'#' * level} {s}")
    d.check("heading", s, page)


def body(d: Doc, s, page, x=MARGIN, y=0.0, width=W - 2 * MARGIN, size=10, leading=13) -> float:
    d.md.append(s)
    d.check("body", s, page)
    return paragraph(d.c, x, y, s, width, size, leading)


def lines(d: Doc, items, page, x, y, size=9.5, leading=12.5, bold_first=False) -> float:
    """Short lines such as an address block; one ground-truth paragraph."""
    for i, s in enumerate(items):
        text(d.c, x, y - i * leading, s, size, bold=bold_first and i == 0)
        d.check("body", s, page)
    d.md.append("\n".join(items))
    return y - len(items) * leading - 8


def grid(d: Doc, rows, widths, page, x=MARGIN, y=0.0, **kw) -> float:
    d.md.append(md_table(rows))
    for row in rows[1:]:
        d.check("table", " ".join(cell for cell in row if cell), page)
    return table(d.c, x, y, rows, widths, **kw)


def qr_slip(d: Doc, page, top, payee, payer, amount, reference):
    """Swiss QR-bill payment part (receipt + payment part) at the bottom of a page."""
    c = d.c
    c.setStrokeColor(HexColor("#000000"))
    c.setDash(2, 2)
    c.line(0, top, W, top)
    c.line(62 * mm, top, 62 * mm, 0)
    c.setDash()
    iban = "CH44 3199 9123 0008 8901 2"
    payload = "\n".join(["SPC", "0200", "1", iban.replace(" ", ""), "K", *payee[:1], *payee[1:3], "", "", "CH",
                         "", "", "", "", "", "", "", amount.replace("'", ""), "CHF", "K", *payer[:1], *payer[1:3],
                         "", "", "CH", "QRR", reference.replace(" ", ""), "", "EPD"])  # fmt: skip
    qr = segno.make(payload, error="m", micro=False)
    buffer = io.BytesIO()
    qr.save(buffer, kind="png", scale=6, border=0)
    buffer.seek(0)
    c.drawImage(raster(Image.open(buffer).convert("RGB")), 67 * mm, top - 68 * mm, 46 * mm, 46 * mm)
    text(c, 5 * mm, top - 10 * mm, "Empfangsschein", 11, bold=True)
    text(c, 67 * mm, top - 10 * mm, "Zahlteil", 11, bold=True)
    d.md += ["## Empfangsschein", "## Zahlteil"]
    d.check("heading", "Empfangsschein", page)
    d.check("heading", "Zahlteil", page)
    for x0 in (5 * mm, 118 * mm):
        y = top - 18 * mm
        for label, value in (("Konto / Zahlbar an", [iban, *payee]), ("Referenz", [reference]),
                             ("Zahlbar durch", payer)):  # fmt: skip
            text(c, x0, y, label, 6.5, bold=True)
            y -= 9
            for v in value:
                text(c, x0, y, v, 8)
                y -= 9.5
            y -= 4
    text(c, 5 * mm, top - 80 * mm, "Währung  Betrag", 6.5, bold=True)
    text(c, 5 * mm, top - 89 * mm, f"CHF      {amount}", 8)
    text(c, 67 * mm, top - 80 * mm, "Währung  Betrag", 6.5, bold=True)
    text(c, 67 * mm, top - 89 * mm, f"CHF      {amount}", 10)
    d.md.append("\n".join(["Konto / Zahlbar an", iban, *payee, "Referenz", reference, "Zahlbar durch", *payer,
                           "Währung Betrag", f"CHF {amount}"]))  # fmt: skip
    for v in (iban, reference, *payee, *payer, f"CHF {amount}"):
        d.check("body", v, page)


# ---------------------------------------------------------------- documents
def doc_lohnausweis(out: Path) -> Doc:
    d = Doc("admin_lohnausweis", out)
    c = d.c
    heading(d, "Lohnausweis – Certificat de salaire – Certificato di salario", 1, y=H - 60, size=13)
    text(c, W - MARGIN - 70, H - 45, "Formular 11", 8)
    y = lines(d, ["A  [x] Lohnausweis", "B  [ ] Rentenbescheinigung"], 1, MARGIN, H - 85)
    y = lines(d, ["C  AHV-Nr. 756.9999.0000.01", "D  Jahr 2025", "E  von 01.01.2025 bis 31.12.2025",
                  "F  [ ] Unentgeltliche Beförderung Wohn-/Arbeitsort",
                  "G  [x] Kantinenverpflegung / Lunch-Checks"], 1, MARGIN, y)  # fmt: skip
    c.rect(W - MARGIN - 170, H - 150, 170, 62)
    lines(d, ["Lea Muster", "Beispielweg 12", "4051 Basel"], 1, W - MARGIN - 160, H - 105, size=10.5)
    rows = [
        ["Ziffer", "Bezeichnung", "CHF"],
        ["1", "Lohn soweit nicht unter Ziffer 2-7 aufzuführen", "84'500.00"],
        ["2.2", "Privatanteil Geschäftswagen", "3'240.00"],
        ["3", "Unregelmässige Leistungen", "6'000.00"],
        ["7", "Andere Leistungen", "450.00"],
        ["8", "Bruttolohn total / Rente", "94'190.00"],
        ["9", "Beiträge AHV/IV/EO/ALV/NBUV", "6'121.10"],
        ["10.1", "Berufliche Vorsorge 2. Säule ordentliche Beiträge", "5'640.00"],
        ["11", "Nettolohn / Rente", "82'428.90"],
        ["13.1.1", "Effektive Spesen Reise, Verpflegung, Übernachtung", "1'230.40"],
        ["15", "Bemerkungen", "Teilzeit 80 %"],
    ]
    y = grid(d, rows, [50, 330, 103], 1, y=y - 10, row_h=20, size=9)
    y = lines(d, ["I  Ort und Datum: Pratteln, 28.01.2026",
                  "Arbeitgeber: Fiktiv Logistik AG, Industriestrasse 5, 4133 Pratteln, Tel. 061 000 00 00"],
              1, MARGIN, y - 10)  # fmt: skip
    draw_hand(d, MARGIN + 300, y - 30, "R. Beispiel", 1, height=22)
    footer(d, 1)
    d.save()
    return d


def doc_stromrechnung(out: Path) -> Doc:
    d = Doc("admin_stromrechnung", out)
    c = d.c
    text(c, MARGIN, H - 50, "Rheinstrom Energie AG", 14, bold=True, color="#0b5a8a")
    text(c, MARGIN, H - 64, "Kundendienst 061 000 11 22 · rheinstrom.example", 8, color="#555555")
    lines(d, ["Lea Muster", "Beispielweg 12", "4051 Basel"], 1, W - MARGIN - 160, H - 100, size=10)
    heading(d, "Rechnung Elektrizität", 1, y=H - 170)
    y = lines(d, ["Rechnungsnummer: 2026-0471183", "Rechnungsdatum: 03.02.2026", "Kundennummer: 300 145 922",
                  "Zählernummer: 79 812 004", "Abrechnungsperiode: 01.01.2025 – 31.12.2025",
                  "Verbrauchsadresse: Beispielweg 12, 4051 Basel"], 1, MARGIN, H - 195)  # fmt: skip
    rows = [
        ["Position", "Menge", "Einheit", "Preis CHF", "Betrag CHF"],
        ["Energie Hochtarif", "1'842", "kWh", "0.1520", "279.98"],
        ["Energie Niedertarif", "1'206", "kWh", "0.1240", "149.54"],
        ["Netznutzung", "3'048", "kWh", "0.0985", "300.23"],
        ["Grundgebühr", "12", "Mt.", "4.50", "54.00"],
        ["Abgaben an das Gemeinwesen", "3'048", "kWh", "0.0230", "70.10"],
        ["Netzzuschlag Bund", "3'048", "kWh", "0.0230", "70.10"],
        ["Total exkl. MWST", None, None, None, "923.95"],
        ["MWST 8.1 %", None, None, None, "74.84"],
        ["Total inkl. MWST", None, None, None, "998.79"],
    ]
    y = grid(d, rows, [190, 60, 50, 80, 103], 1, y=y - 6, zebra="#f2f6f9")
    heading(d, "Ihr Verbrauch der letzten 12 Monate (kWh)", 1, y=y - 10, size=11, level=2)
    months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    values = [352, 318, 290, 236, 201, 178, 169, 174, 205, 248, 301, 376]
    base, x0 = y - 160, MARGIN + 20
    c.setFillColor(HexColor("#0b5a8a"))
    for i, (m, v) in enumerate(zip(months, values, strict=True)):
        c.rect(x0 + i * 36, base, 22, v * 0.32, stroke=0, fill=1)
        text(c, x0 + i * 36 + 2, base - 12, m, 7.5)
        text(c, x0 + i * 36, base + v * 0.32 + 3, str(v), 7)
    for m in months:
        d.check("chart_text", m, 1)
    footer(d, 1)
    c.showPage()
    heading(d, "Zahlung", 2, y=H - 60, size=13, level=2)
    body(d, "Bitte überweisen Sie den Betrag von CHF 998.79 bis zum 05.03.2026 mit dem untenstehenden "
            "Zahlteil. Bei Fragen zur Rechnung erreichen Sie uns von Montag bis Freitag.", 2, y=H - 85)  # fmt: skip
    qr_slip(d, 2, 105 * mm, ["Rheinstrom Energie AG", "Postfach", "4002 Basel"],
            ["Lea Muster", "Beispielweg 12", "4051 Basel"], "998.79", "21 00000 00003 13947 14300 09017")  # fmt: skip
    footer(d, 2)
    d.save()
    return d


def doc_rechnung_qr(out: Path) -> Doc:
    d = Doc("admin_rechnung_qr", out)
    lines(d, ["Sanitär Beispiel GmbH", "Werkstrasse 7", "4123 Allschwil", "MWST CHE-000.111.222"], 1,
          MARGIN, H - 50, bold_first=True)  # fmt: skip
    lines(d, ["Lea Muster", "Beispielweg 12", "4051 Basel"], 1, W - MARGIN - 160, H - 110, size=10)
    heading(d, "Rechnung Nr. 26-0193", 1, y=H - 180)
    y = lines(d, ["Datum: 17.02.2026", "Auftrag: Ersatz Mischbatterie Küche"], 1, MARGIN, H - 200)
    rows = [
        ["Position", "Menge", "Ansatz CHF", "Betrag CHF"],
        ["Arbeitsstunden Monteur", "6.5 Std.", "115.00", "747.50"],
        ["Mischbatterie Küche", "1 Stk.", "289.00", "289.00"],
        ["Kleinmaterial", "pauschal", "35.00", "35.00"],
        ["Fahrzeugpauschale", "2 x", "45.00", "90.00"],
        ["Zwischentotal", None, None, "1'161.50"],
        ["MWST 8.1 %", None, None, "94.08"],
        ["Total", None, None, "1'255.58"],
    ]
    y = grid(d, rows, [210, 80, 90, 103], 1, y=y - 6, grid=False, header_fill="#e8e8e8")
    body(d, "Zahlbar innert 30 Tagen netto. Wir danken für Ihren Auftrag.", 1, y=y)
    qr_slip(d, 1, 105 * mm, ["Sanitär Beispiel GmbH", "Werkstrasse 7", "4123 Allschwil"],
            ["Lea Muster", "Beispielweg 12", "4051 Basel"], "1'255.58", "00 00000 00000 00026 01930 00008")  # fmt: skip
    footer(d, 1)
    d.save()
    return d


def doc_kassenbon(out: Path) -> Doc:
    d = Doc("admin_kassenbon", out)
    c = d.c
    width, height = 80 * mm, 190 * mm
    c.setPageSize((width, height))
    y = height - 30
    entries = [
        ("QUARTIERLADEN AM BACH", True), ("Bachgasse 9, 4000 Musterstadt", False), ("", False),
        ("Vollmilch 1l               1.75", False), ("Ruchbrot 500g              3.20", False),
        ("Äpfel Gala 1.2kg           4.68", False), ("Emmentaler 250g            5.90", False),
        ("Mineralwasser 6x1.5l       4.50", False), ("Bananen 0.8kg              2.32", False),
        ("", False), ("TOTAL CHF                 22.35", True), ("Debitkarte                22.35", False),
        ("", False), ("MWST   Satz   Netto   MWST", False), ("       2.6%   21.78   0.57", False),
        ("MWST-Nr. CHE-000.333.444 MWST", False), ("14.03.2026 17:42  Kasse 2  Bon 4711", False),
        ("Vielen Dank für Ihren Einkauf", False),
    ]  # fmt: skip
    for s, bold in entries:
        if s:
            c.setFillColor(HexColor("#000000"))
            c.setFont("Mono", 8.5 if bold else 7.6)
            c.drawCentredString(width / 2, y, s) if bold or "Vielen" in s or "Bach" in s else c.drawString(10, y, s)
            d.check("body", " ".join(s.split()), 1)
        y -= 13
    d.md.append("\n".join(" ".join(s.split()) for s, _ in entries if s))
    footer(d, 1, width)
    d.save()
    return d


def doc_verfuegung(out: Path) -> Doc:
    d = Doc("admin_verfuegung", out)
    c = d.c
    lines(d, ["Stadt Musterstadt", "Amt für Bewilligungen", "Marktplatz 1, 4000 Musterstadt"], 1, MARGIN, H - 50,
          bold_first=True)  # fmt: skip
    lines(d, ["Einschreiben", "Café Beispiel GmbH", "Herr Max Beispiel", "Seestrasse 3", "4000 Musterstadt"], 1,
          W - MARGIN - 170, H - 120, size=10)  # fmt: skip
    text(c, MARGIN, H - 215, "Musterstadt, 9. März 2026", 10)
    d.md.append("Musterstadt, 9. März 2026")
    d.check("body", "Musterstadt, 9. März 2026", 1)
    heading(d, "Verfügung betreffend Bewilligung für ein Strassencafé", 1, y=H - 245, size=13)
    y = H - 270
    heading(d, "Sachverhalt", 1, y=y, size=11, level=2)
    y = body(d, "Mit Gesuch vom 12. Januar 2026 beantragt die Café Beispiel GmbH die Bewilligung, auf der "
                "Allmend vor der Liegenschaft Seestrasse 3 sechs Tische mit 24 Sitzplätzen aufzustellen. Das "
                "Gesuch wurde vom 2. bis 23. Februar 2026 öffentlich aufgelegt.", 1, y=y - 18)  # fmt: skip
    heading(d, "Erwägungen", 1, y=y, size=11, level=2)
    for i, s in enumerate([
        "Die Nutzung der Allmend für Gastwirtschaftszwecke ist bewilligungspflichtig. Zuständig ist das Amt "
        "für Bewilligungen.",
        "Während der Auflage ging eine Einsprache ein. Sie richtet sich gegen den Betrieb nach 22 Uhr. Dem "
        "Anliegen wird mit einer Auflage zu den Betriebszeiten Rechnung getragen.",
        "Die Durchgangsbreite für Fussgängerinnen und Fussgänger von mindestens 2.0 Metern bleibt gewahrt.",
    ], 1):  # fmt: skip
        y = body(d, f"{i}. {s}", 1, y=y - (18 if i == 1 else 0))
    footer(d, 1)
    c.showPage()
    heading(d, "Verfügung", 2, y=H - 60, size=11, level=2)
    y = H - 80
    for i, s in enumerate(["Die Bewilligung für sechs Tische mit 24 Sitzplätzen wird befristet bis 31. Oktober "
                           "2026 erteilt.", "Der Betrieb ist täglich bis 22 Uhr gestattet.",
                           "Die Gebühr beträgt CHF 380.00 und ist innert 30 Tagen zu bezahlen."], 1):  # fmt: skip
        y = body(d, f"{i}. {s}", 2, y=y)
    heading(d, "Rechtsmittelbelehrung", 2, y=y - 6, size=11, level=2)
    y = body(d, "Gegen diese Verfügung kann innert 10 Tagen seit Eröffnung beim Departement Rekurs angemeldet "
                "werden. Die Rekursbegründung ist innert 30 Tagen nach der Anmeldung einzureichen.", 2,
             y=y - 24)  # fmt: skip
    y = lines(d, ["Freundliche Grüsse", "Amt für Bewilligungen"], 2, MARGIN, y - 10)
    draw_hand(d, MARGIN, y - 30, "A. Muster", 2, height=26, font="Kalam.ttf")
    lines(d, ["Anna Muster, Leiterin"], 2, MARGIN, y - 45)
    stamp = Image.new("RGBA", (520, 200), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rounded_rectangle((8, 8, 512, 192), 20, outline=(170, 30, 40, 220), width=8)
    sf = ImageFont.truetype(str(BOLD), 44)
    sd.text((40, 40), "AMT FÜR", font=sf, fill=(170, 30, 40, 220))
    sd.text((40, 100), "BEWILLIGUNGEN", font=sf, fill=(170, 30, 40, 220))
    c.drawImage(raster(stamp.rotate(-8, expand=True)), MARGIN + 230, y - 70, 150, 64, mask="auto")
    d.check("bitmap_text", "AMT FÜR BEWILLIGUNGEN", 2)
    footer(d, 2)
    d.save()
    return d


def doc_formular_hand(out: Path) -> Doc:
    """A printed application form filled in by hand; only exists on paper, so no clean version."""
    d = Doc("admin_formular_hand", out)
    c = d.c
    heading(d, "Gesuch um Anwohnerparkkarte", 1, y=H - 70)
    text(c, MARGIN, H - 88, "Bitte in Blockschrift ausfüllen", 9, color="#444444")
    d.md.append("Bitte in Blockschrift ausfüllen")
    d.check("body", "Bitte in Blockschrift ausfüllen", 1)
    fields = [("Name", "Muster"), ("Vorname", "Lea"), ("Geburtsdatum", "14.07.1988"),
              ("Strasse, Nr.", "Beispielweg 12"), ("PLZ, Ort", "4051 Basel"), ("Telefon", "079 000 12 34"),
              ("Kontrollschild", "BS 123 456"), ("Fahrzeugmarke", "VW Golf"), ("Gültig ab", "1. April 2026")]  # fmt: skip
    y = H - 130
    for i, (label, value) in enumerate(fields):
        text(c, MARGIN, y, label, 10)
        d.check("body", label, 1)
        c.setStrokeColor(HexColor("#888888"))
        c.line(MARGIN + 120, y - 3, W - MARGIN, y - 3)
        draw_hand(d, MARGIN + 130, y - 4, value, 1, height=28, font="Caveat.ttf" if i % 2 else "Kalam.ttf")
        d.md.append(f"{label} {value}")
        y -= 38
    for label, ticked in (("Anwohnerparkkarte", True), ("Besucherkarte", False), ("Gewerbeparkkarte", False)):
        c.rect(MARGIN, y - 2, 10, 10)
        if ticked:
            draw_hand(d, MARGIN - 2, y - 4, "x", 1, height=16)
        text(c, MARGIN + 18, y, label, 10)
        d.check("body", label, 1)
        d.md.append(f"[{'x' if ticked else ' '}] {label}")
        y -= 22
    text(c, MARGIN, y - 30, "Ort, Datum", 10)
    d.check("body", "Ort, Datum", 1)
    draw_hand(d, MARGIN + 130, y - 34, "Basel, 20.03.2026", 1, height=28)
    text(c, MARGIN, y - 70, "Unterschrift", 10)
    d.check("body", "Unterschrift", 1)
    draw_hand(d, MARGIN + 130, y - 76, "L. Muster", 1, height=26, font="Kalam.ttf")
    d.md += ["Ort, Datum Basel, 20.03.2026", "Unterschrift L. Muster"]
    footer(d, 1)
    d.save()
    return d


def doc_krankenkasse(out: Path) -> Doc:
    d = Doc("admin_krankenkasse", out)
    lines(d, ["Gesundkasse Beispiel", "Postfach, 4000 Musterstadt"], 1, MARGIN, H - 50, bold_first=True)
    lines(d, ["Lea Muster", "Beispielweg 12", "4051 Basel"], 1, W - MARGIN - 160, H - 100, size=10)
    heading(d, "Leistungsabrechnung vom 31.03.2026", 1, y=H - 170, size=13)
    y = lines(d, ["Versichertennummer: 80756 99990 00001", "Grundversicherung, Franchise CHF 1'500"], 1,
              MARGIN, H - 190)  # fmt: skip
    rows = [["Datum", "Leistungserbringer", "Leistung", "Betrag", "Franchise", "Selbstbehalt", "Vergütung"]]
    data = [
        ("08.01.2026", "Praxis Dr. A. Beispiel", "Konsultation", "142.30"),
        ("08.01.2026", "Apotheke am Platz", "Medikamente", "48.65"),
        ("22.01.2026", "Labor Muster AG", "Laboranalysen", "96.40"),
        ("05.02.2026", "Praxis Dr. A. Beispiel", "Konsultation", "118.90"),
        ("05.02.2026", "Physiotherapie Beispiel", "Physiotherapie 9 Sitzungen", "486.00"),
        ("19.02.2026", "Spital Musterstadt", "Ambulante Behandlung", "1'274.55"),
        ("03.03.2026", "Apotheke am Platz", "Medikamente", "63.20"),
        ("17.03.2026", "Praxis Dr. A. Beispiel", "Konsultation", "142.30"),
    ]
    remaining = 1500.0
    for date, provider, service, amount in data:
        value = float(amount.replace("'", ""))
        franchise = min(value, remaining)
        remaining -= franchise
        share = round((value - franchise) * 0.1, 2)
        paid = value - franchise - share
        rows.append([date, provider, service, amount, f"{franchise:,.2f}".replace(",", "'"), f"{share:.2f}",
                     f"{paid:,.2f}".replace(",", "'")])  # fmt: skip
    y = grid(d, rows, [52, 110, 112, 52, 52, 55, 50], 1, y=y - 6, size=7.2, row_h=15)
    body(d, "Die Vergütung wird innert 10 Tagen auf Ihr Konto überwiesen. Einwände gegen diese Abrechnung "
            "richten Sie bitte innert 30 Tagen schriftlich an uns.", 1, y=y)  # fmt: skip
    footer(d, 1)
    d.save()
    return d


# ---------------------------------------------------------------- degraded copies
def render(pdf: Path, dpi: int) -> list[Image.Image]:
    document = pypdfium2.PdfDocument(str(pdf))
    pages = [page.render(scale=dpi / 72).to_pil().convert("L") for page in document]
    document.close()
    return pages


def degrade(img: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, dtype=float)
    if mode == "scan":
        a = 245 - (245 - 15) * (1 - a / 255)
        img = Image.fromarray(a.clip(0, 255).astype("uint8")).rotate(rng.uniform(-0.9, 0.9), fillcolor=245,
                                                                      resample=Image.BICUBIC)  # fmt: skip
        a = np.asarray(img, dtype=float) + rng.normal(0, 7, a.shape)
        return Image.fromarray(a.clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.5))
    if mode == "fax":
        small = img.resize((img.width, img.height // 2), Image.BILINEAR).resize(img.size, Image.NEAREST)
        a = np.asarray(small, dtype=float) + rng.normal(0, 25, a.shape)
        b = np.where(a < 150, 0, 255).astype("uint8")
        speckle = rng.random(b.shape) < 0.002
        b[speckle] = 255 - b[speckle]
        return Image.fromarray(b).rotate(0.4, fillcolor=255)
    if mode == "thermal":
        fade = np.linspace(0.55, 0.85, a.shape[1])[None, :]
        a = 238 - (238 - 110) * (1 - a / 255) * fade + rng.normal(0, 6, a.shape)
        return Image.fromarray(a.clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.7))
    raise ValueError(mode)


def photo(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """The page photographed on a table: background, perspective, uneven light."""
    page = img.convert("RGB")
    canvas_w, canvas_h = int(page.width * 1.35), int(page.height * 1.25)
    background = rng.normal(0, 10, (canvas_h, canvas_w, 1)) + np.array([150, 118, 86])
    canvas = Image.fromarray(background.clip(0, 255).astype("uint8"))
    canvas.paste(page, ((canvas_w - page.width) // 2, (canvas_h - page.height) // 2))
    w, h = canvas.size
    jitter = [rng.uniform(0.02, 0.07) for _ in range(8)]
    corners = [(w * jitter[0], h * jitter[1]), (w * (1 - jitter[2]), h * jitter[3]),
               (w * (1 - jitter[4]), h * (1 - jitter[5])), (w * jitter[6], h * (1 - jitter[7]))]  # fmt: skip
    target = [(0, 0), (w, 0), (w, h), (0, h)]
    matrix = []
    for (x, y), (u, v) in zip(target, corners, strict=True):
        matrix += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
    coefficients = np.linalg.solve(np.array(matrix), np.array(corners).reshape(8))
    warped = canvas.transform(canvas.size, Image.PERSPECTIVE, coefficients, Image.BICUBIC)
    light = np.linspace(1.08, 0.72, w)[None, :, None] * np.linspace(1.0, 0.85, h)[:, None, None]
    shaded = (np.asarray(warped, dtype=float) * light).clip(0, 255).astype("uint8")
    result = Image.fromarray(shaded).filter(ImageFilter.GaussianBlur(1.0))
    return result.resize((result.width * 2 // 3, result.height * 2 // 3), Image.LANCZOS)


def write_variant(d: Doc, src: Path, out: Path, mode: str, seed: int, keep=("furniture", "handwriting")) -> None:
    """A degraded copy of d with the same ground truth; text categories become scan_text/photo_text."""
    rng = np.random.default_rng(seed)
    name = f"{d.name}_{mode}"
    category = "photo_text" if mode == "photo" else "scan_text"
    checks = [{**chk, "category": chk["category"] if chk["category"] in keep else category} for chk in d.checks]
    if mode == "photo":
        photo(render(src, 200)[0], rng).save(out / f"{name}.jpg", quality=82)
    else:
        source = pypdfium2.PdfDocument(str(src))
        sizes = [page.get_size() for page in source]
        source.close()
        c = Canvas(str(out / f"{name}.pdf"))
        c.setTitle(name)
        for img, (width, height) in zip(render(src, 300 if mode == "thermal" else 200), sizes, strict=True):
            c.setPageSize((width, height))
            degraded = degrade(img, "scan" if mode == "scan" else mode, rng)
            c.drawImage(raster(degraded, "JPEG" if mode != "fax" else "PNG", 72), 0, 0, width, height)
            c.showPage()
        c.save()
    (out / f"{name}.gt.md").write_text("\n\n".join(d.md).strip() + "\n")
    (out / f"{name}.checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # name -> (builder, keep the clean PDF, degraded variants)
    plan = {
        "lohnausweis": (doc_lohnausweis, True, ["scan"]),
        "stromrechnung": (doc_stromrechnung, True, ["scan"]),
        "rechnung_qr": (doc_rechnung_qr, True, ["fax", "photo"]),
        "kassenbon": (doc_kassenbon, True, ["thermal", "photo"]),
        "verfuegung": (doc_verfuegung, True, ["scan", "fax"]),
        "formular_hand": (doc_formular_hand, False, ["scan", "photo"]),
        "krankenkasse": (doc_krankenkasse, True, ["scan"]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for seed, (build, keep_clean, variants) in enumerate(plan.values()):
            d = build(Path(tmp))
            src = Path(tmp) / f"{d.name}.pdf"
            for mode in variants:
                write_variant(d, src, args.out, mode, seed)
            if keep_clean:
                for suffix in (".pdf", ".gt.md", ".checks.json"):
                    shutil.copy(Path(tmp) / f"{d.name}{suffix}", args.out / f"{d.name}{suffix}")
            print("wrote", d.name, "clean" if keep_clean else "", *variants)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the two TRAPs Word reports (.docx) from their Markdown sources in the LBT house style.

    python3 scripts/build_word_reports.py [--md-dir docs/reports/word]

Input  : docs/reports/word/TRAPs_實作方法.md, TRAPs_執行成果.md (+ figures/*.jpg)
Output : docs/reports/word/TRAPs_實作方法.docx, TRAPs_執行成果.docx

Dialect handled (the one used by Lagrangian-Ensemble-Backtracking's report sources):
  # / ## / ### / ####      headings                       (Heading 1-4 styles, bold, 標楷體)
  　　text                    body paragraph with full-width indent
  〔編輯說明：…〕              italic editorial note
  **表甲-N  標題**           table caption (bold, centred, above table);  圖 captions sit below the figure
  | a | b |                  pipe tables (header shaded E7E6E6, centred cells, repeated header row)
  資料來源：…                source line under each table
  $$ … \\qquad (n) $$        display equation  -> borderless 3-column table [blank | OMML | (n)] as in LBT
  $…$                        inline equation   -> OMML (m:oMath), converted from TeX by pandoc
  ![](path){width=6.3in}     centred picture

Formatting replicated from the LBT .docx files (see CFG): A4, margins 1417 twips, Times New Roman for Latin and
標楷體 for East Asian text set on every run, grey header row, compact table cells.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, Twips
from lxml import etree

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

LATIN, EAST = "Times New Roman", "標楷體"

CFG = {
    # LBT 實作方法.docx: body 12 pt, line 1.5 (360), after 6 pt; tables 10 pt; caption 11 pt bold
    "method": dict(body=12, line=360, after=120, heads={1: 18, 2: 16, 3: 14, 4: 12.5}, hafter={1: 160, 2: 160, 3: 160, 4: 80},
                   hbefore={1: 0, 2: 0, 3: 120, 4: 0}, cap=11, src=10, src_after=200, note=10.5,
                   table_pt=lambda ncol: 10),
    # LBT 執行成果.docx: body 14 pt, line 1.3 (312), after 6 pt, headings bold 14-16 pt, tables 14 pt full width
    "results": dict(body=14, line=312, after=120, heads={1: 16, 2: 14, 3: 14, 4: 14}, hafter={1: 120, 2: 120, 3: 120, 4: 120},
                    hbefore={1: 240, 2: 240, 3: 240, 4: 240}, cap=14, src=14, src_after=120, note=12,
                    table_pt=lambda ncol: 14 if ncol <= 4 else (11 if ncol <= 6 else 10.5)),
}
PAGE_TEXT_TWIPS = 11906 - 2 * 1417  # 9072


# ----------------------------------------------------------------------------- markdown parsing
CAP_RE = re.compile(r"^\*\*((?:表|圖)[甲乙]-\d+)  (.*)\*\*$")
IMG_RE = re.compile(r"^!\[\]\((.+?)\)\{width=([\d.]+)in\}$")
EQNUM_RE = re.compile(r"\\qquad \((\d+)\)\s*$")


def parse_blocks(md: str):
    blocks = []
    for chunk in re.split(r"\n\s*\n", md.strip()):
        chunk = chunk.rstrip()
        if not chunk:
            continue
        lines = chunk.split("\n")
        first = lines[0]
        if chunk.startswith("$$") and chunk.endswith("$$"):
            tex = chunk[2:-2].strip()
            mnum = EQNUM_RE.search(tex)
            num = mnum.group(1) if mnum else ""
            if mnum:
                tex = tex[: mnum.start()].rstrip()
            blocks.append(("eq", tex, num))
        elif re.match(r"^#{1,4} ", first):
            level = len(first.split(" ")[0])
            blocks.append(("h", level, first[level + 1:].strip()))
        elif first.startswith("|"):
            rows = []
            for ln in lines:
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                rows.append(cells)
            header, body = rows[0], [r for r in rows[2:]]
            blocks.append(("table", header, body))
        elif CAP_RE.match(chunk):
            mm = CAP_RE.match(chunk)
            blocks.append(("caption", mm.group(1), mm.group(2)))
        elif IMG_RE.match(chunk):
            mm = IMG_RE.match(chunk)
            blocks.append(("img", mm.group(1), float(mm.group(2))))
        elif chunk.startswith("〔"):
            blocks.append(("note", chunk))
        elif chunk.startswith("資料來源："):
            blocks.append(("source", chunk))
        else:
            blocks.append(("p", " ".join(lines)))
    return blocks


INLINE_RE = re.compile(r"\$([^$\n]+?)\$|\*\*(.+?)\*\*")


def tokenize(text: str):
    """-> list of ('t', str) | ('b', str) | ('m', tex)"""
    out, pos = [], 0
    for mt in INLINE_RE.finditer(text):
        if mt.start() > pos:
            out.append(("t", text[pos:mt.start()]))
        if mt.group(1) is not None:
            out.append(("m", mt.group(1).strip()))
        else:
            out.append(("b", mt.group(2)))
        pos = mt.end()
    if pos < len(text):
        out.append(("t", text[pos:]))
    return out


def collect_math(blocks):
    seen, order = {}, []

    def add(tex, display):
        key = (tex, display)
        if key not in seen:
            seen[key] = len(order)
            order.append(key)

    def scan(text):
        for kind, val in tokenize(text):
            if kind == "m":
                add(val, False)
            elif kind == "b":
                scan(val)

    for b in blocks:
        if b[0] == "eq":
            add(b[1], True)
        elif b[0] in ("p", "note", "source"):
            scan(b[1])
        elif b[0] == "caption":
            scan(b[2])
        elif b[0] == "table":
            for c in b[1]:
                scan(c)
            for r in b[2]:
                for c in r:
                    scan(c)
    return seen, order


def tex_to_omml(order):
    """Convert every TeX string with one pandoc call; return {key: oMath element}."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.md"
        parts = []
        for tex, display in order:
            parts.append(("$$" + tex + "$$") if display else ("$" + tex + "$"))
        src.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        out = Path(td) / "m.docx"
        subprocess.run(["pandoc", str(src), "-f", "markdown+tex_math_dollars", "-o", str(out)], check=True)
        with zipfile.ZipFile(out) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    maths = root.findall(".//{%s}oMath" % M_NS)
    if len(maths) != len(order):
        raise RuntimeError(f"pandoc produced {len(maths)} equations for {len(order)} inputs")
    return {key: m for key, m in zip(order, maths)}


# ----------------------------------------------------------------------------- docx helpers
def set_run_font(run, size, bold=False, italic=False):
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    rf.set(qn("w:ascii"), LATIN)
    rf.set(qn("w:hAnsi"), LATIN)
    rf.set(qn("w:eastAsia"), EAST)
    rf.set(qn("w:cs"), LATIN)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic


def para_fmt(p, *, before=0, after=120, line=None, align=None, keep_next=False, keep_lines=False):
    pf = p.paragraph_format
    pf.space_before = Twips(before)
    pf.space_after = Twips(after)
    if line:
        ppr = p._p.get_or_add_pPr()
        sp = ppr.find(qn("w:spacing"))
        sp.set(qn("w:line"), str(line))
        sp.set(qn("w:lineRule"), "auto")
    if align is not None:
        p.alignment = align
    if keep_next:
        pf.keep_with_next = True
    if keep_lines:
        pf.keep_together = True


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
               "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook"]
TRPR_ORDER = ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit", "trHeight", "tblHeader",
              "tblCellSpacing", "jc", "hidden"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar", "textDirection",
              "tcFitText", "vAlign", "hideMark"]


def fix_order(el, order):
    """Re-sequence child elements into WordprocessingML schema order (Word is strict about it)."""
    rank = {qn("w:" + n): i for i, n in enumerate(order)}
    kids = list(el)
    kids.sort(key=lambda k: rank.get(k.tag, len(order)))
    for k in kids:
        el.remove(k)
    for k in kids:
        el.append(k)


class Builder:
    def __init__(self, kind: str, md_dir: Path, omml):
        self.cfg = CFG[kind]
        self.kind = kind
        self.md_dir = md_dir
        self.omml = omml
        self.doc = Document()
        self._setup()
        self.img_count = 0

    # -- document set-up ---------------------------------------------------------------------
    def _setup(self):
        d = self.doc
        s = d.sections[0]
        s.page_width, s.page_height = Twips(11906), Twips(16838)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, side, Twips(1417))
        s.header_distance = s.footer_distance = Twips(708)
        # styles: Normal and Heading 1-4 in TNR / 標楷體
        st = d.styles
        normal = st["Normal"]
        normal.font.name = LATIN
        normal.font.size = Pt(self.cfg["body"])
        self._style_fonts(normal)
        for lvl in (1, 2, 3, 4):
            hs = st[f"Heading {lvl}"]
            hs.font.name = LATIN
            hs.font.size = Pt(self.cfg["heads"][lvl])
            hs.font.bold = True
            hs.font.italic = False
            hs.font.color.rgb = None
            self._style_fonts(hs)
            hs.paragraph_format.keep_with_next = True
            hs.paragraph_format.space_before = Twips(self.cfg["hbefore"][lvl])
            hs.paragraph_format.space_after = Twips(self.cfg["hafter"][lvl])
        # document language
        rpr_default = d.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
        lang = rpr_default.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            rpr_default.append(lang)
        lang.set(qn("w:val"), "en-US")
        lang.set(qn("w:eastAsia"), "zh-TW")
        d.core_properties.author = ""
        d.core_properties.title = "TRAPs 報告章節"

    @staticmethod
    def _style_fonts(style):
        rpr = style.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for a in list(rf.attrib):
            del rf.attrib[a]
        rf.set(qn("w:ascii"), LATIN)
        rf.set(qn("w:hAnsi"), LATIN)
        rf.set(qn("w:eastAsia"), EAST)
        rf.set(qn("w:cs"), LATIN)
        rpr.append(OxmlElement("w:color")) if False else None

    # -- inline content ----------------------------------------------------------------------
    def add_inline(self, p, text, size, bold=False, italic=False):
        for kind, val in tokenize(text):
            if kind == "t":
                for i, part in enumerate(val.split("<br>")):
                    if i:
                        p.add_run().add_break()
                    r = p.add_run(part)
                    set_run_font(r, size, bold, italic)
            elif kind == "b":
                self.add_inline(p, val, size, True, italic)
            else:
                p._p.append(self._sized_omml(self.omml[(val, False)], size))

    @staticmethod
    def _sized_omml(node, size):
        """Copy of an oMath element whose math runs carry an explicit font size (Word otherwise uses its own default)."""
        node = copy.deepcopy(node)
        for r in node.iter("{%s}r" % M_NS):
            if r.find("{%s}rPr" % W_NS) is None:
                rpr = etree.SubElement(r, "{%s}rPr" % W_NS)
                rf = etree.SubElement(rpr, "{%s}rFonts" % W_NS)
                rf.set("{%s}ascii" % W_NS, "Cambria Math")
                rf.set("{%s}hAnsi" % W_NS, "Cambria Math")
                sz = etree.SubElement(rpr, "{%s}sz" % W_NS)
                sz.set("{%s}val" % W_NS, str(int(round(size * 2))))
                # schema order: m:rPr, w:rPr, m:t
                r.remove(rpr)
                mrpr = r.find("{%s}rPr" % M_NS)
                if mrpr is not None:
                    mrpr.addnext(rpr)
                else:
                    r.insert(0, rpr)
        return node

    # -- blocks ------------------------------------------------------------------------------
    def heading(self, level, text):
        p = self.doc.add_paragraph(style=f"Heading {level}")
        r = p.add_run(text)
        set_run_font(r, self.cfg["heads"][level], bold=True)
        r.font.color.rgb = None
        para_fmt(p, before=self.cfg["hbefore"][level], after=self.cfg["hafter"][level], line=360 if self.kind == "method" else None, keep_next=True)

    def body(self, text):
        p = self.doc.add_paragraph()
        self.add_inline(p, text, self.cfg["body"])
        para_fmt(p, after=self.cfg["after"], line=self.cfg["line"])
        return p

    def note(self, text):
        p = self.doc.add_paragraph()
        self.add_inline(p, text, self.cfg["note"], italic=True)
        para_fmt(p, after=120, line=self.cfg["line"])

    def caption(self, label, title, *, above):
        p = self.doc.add_paragraph()
        self.add_inline(p, f"{label}  {title}", self.cfg["cap"], bold=True)
        para_fmt(p, after=60 if above else 200, before=0 if above else 40, line=360 if self.kind == "method" else self.cfg["line"],
                 align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=above)
        return p

    def source(self, text):
        p = self.doc.add_paragraph()
        self.add_inline(p, text, self.cfg["src"])
        para_fmt(p, after=self.cfg["src_after"], line=360 if self.kind == "method" else self.cfg["line"])

    def picture(self, path, width_in):
        p = self.doc.add_paragraph()
        r = p.add_run()
        width = min(width_in, PAGE_TEXT_TWIPS / 1440.0)
        r.add_picture(str(self.md_dir / path), width=Inches(width))
        para_fmt(p, after=40, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
        self.img_count += 1

    def equation(self, tex, num):
        tbl = self.doc.add_table(rows=1, cols=3)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        tblPr = tbl._tbl.tblPr
        for tag in ("w:tblStyle",):
            el = tblPr.find(qn(tag))
            if el is not None:
                tblPr.remove(el)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "nil")
            borders.append(e)
        tblPr.append(borders)
        widths = [907, 7258, 907]
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        row = tbl.rows[0]
        trpr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trpr.append(cs)
        for c, w in zip(row.cells, widths):
            c.width = Twips(w)
            tcpr = c._tc.get_or_add_tcPr()
            va = OxmlElement("w:vAlign")
            va.set(qn("w:val"), "center")
            tcpr.append(va)
        mid = row.cells[1].paragraphs[0]
        omath = self._sized_omml(self.omml[(tex, True)], self.cfg['body'])
        para = OxmlElement("m:oMathPara")
        ppr = OxmlElement("m:oMathParaPr")
        jc = OxmlElement("m:jc")
        jc.set(qn("m:val"), "center")
        ppr.append(jc)
        para.append(ppr)
        para.append(omath)
        mid._p.append(para)
        para_fmt(mid, after=40, line=240)
        right = row.cells[2].paragraphs[0]
        r = right.add_run(f"({num})")
        set_run_font(r, self.cfg["body"])
        right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        para_fmt(right, after=40, line=240, align=WD_ALIGN_PARAGRAPH.RIGHT)
        fix_order(tblPr, TBLPR_ORDER)
        fix_order(trpr, TRPR_ORDER)
        for c in row.cells:
            fix_order(c._tc.get_or_add_tcPr(), TCPR_ORDER)
        # LBT puts an empty paragraph (40 after, line 144) after each equation table to separate blocks
        sp = self.doc.add_paragraph()
        para_fmt(sp, after=40, line=144)

    def table(self, header, rows):
        ncol = len(header)
        pt = self.cfg["table_pt"](ncol)
        tbl = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        tblPr = tbl._tbl.tblPr
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        # column widths ~ content, summing to the text width
        def disp(s):
            s = re.sub(r"\$[^$]*\$", "x", s).replace("**", "")
            return max(sum(2 if ord(ch) > 0x2E80 else 1 for ch in part) for part in s.split("<br>"))

        weights = []
        for j in range(ncol):
            lens = [disp(header[j])] + [disp(r[j]) for r in rows]
            weights.append(min(max(max(lens), 9), 22) + 8)
        total = sum(weights)
        widths = [int(PAGE_TEXT_TWIPS * w / total) for w in weights]
        floor = 1000
        for _ in range(5):  # enforce a minimum column width, taking the excess from the widest columns
            short = [j for j, w in enumerate(widths) if w < floor]
            if not short:
                break
            need = sum(floor - widths[j] for j in short)
            for j in short:
                widths[j] = floor
            donors = [j for j, w in enumerate(widths) if w > floor * 1.6]
            tot = sum(widths[j] for j in donors) or 1
            for j in donors:
                widths[j] -= int(need * widths[j] / tot)
        widths[-1] += PAGE_TEXT_TWIPS - sum(widths)
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        keep_all = len(rows) <= 14
        for i in range(1 + len(rows)):
            cells = header if i == 0 else rows[i - 1]
            tr = tbl.rows[i]
            trpr = tr._tr.get_or_add_trPr()
            if i == 0:
                th = OxmlElement("w:tblHeader")
                th.set(qn("w:val"), "true")
                trpr.append(th)
            cs = OxmlElement("w:cantSplit")
            cs.set(qn("w:val"), "true")
            trpr.append(cs)
            for j, txt in enumerate(cells):
                c = tr.cells[j]
                c.width = Twips(widths[j])
                if i == 0:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E7E6E6")
                    c._tc.get_or_add_tcPr().append(shd)
                va = OxmlElement("w:vAlign")
                va.set(qn("w:val"), "center")
                c._tc.get_or_add_tcPr().append(va)
                p = c.paragraphs[0]
                self.add_inline(p, txt, pt, bold=(i == 0))
                para_fmt(p, before=20, after=20, line=240, align=WD_ALIGN_PARAGRAPH.CENTER,
                         keep_next=(keep_all and i < len(rows)) or (not keep_all and i < 2))
            fix_order(trpr, TRPR_ORDER)
            for c in tr.cells:
                fix_order(c._tc.get_or_add_tcPr(), TCPR_ORDER)
        fix_order(tblPr, TBLPR_ORDER)

    # -- driver -------------------------------------------------------------------------------
    def build(self, blocks):
        pending_caption = None
        for b in blocks:
            kind = b[0]
            if kind == "h":
                self.heading(b[1], b[2])
            elif kind == "p":
                self.body(b[1])
            elif kind == "note":
                self.note(b[1])
            elif kind == "eq":
                self.equation(b[1], b[2])
            elif kind == "caption":
                label, title = b[1], b[2]
                self.caption(label, title, above=label.startswith("表"))
            elif kind == "table":
                self.table(b[1], b[2])
            elif kind == "img":
                self.picture(b[1], b[2])
            elif kind == "source":
                self.source(b[1])
        return self.doc


def build_one(kind, md_path: Path, out_path: Path):
    blocks = parse_blocks(md_path.read_text(encoding="utf-8"))
    seen, order = collect_math(blocks)
    omml = tex_to_omml(order) if order else {}
    b = Builder(kind, md_path.parent, omml)
    doc = b.build(blocks)
    doc.save(str(out_path))
    return len(blocks), len(order), b.img_count


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-dir", default=str(Path(__file__).resolve().parents[1] / "docs" / "reports" / "word"))
    args = ap.parse_args(argv)
    d = Path(args.md_dir)
    for kind, stem in (("method", "TRAPs_實作方法"), ("results", "TRAPs_執行成果")):
        nb, nm, ni = build_one(kind, d / f"{stem}.md", d / f"{stem}.docx")
        print(f"{stem}.docx: {nb} blocks, {nm} equations, {ni} pictures, {(d / (stem + '.docx')).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())

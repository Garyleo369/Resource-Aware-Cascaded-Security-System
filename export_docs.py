import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def markdown_to_docx(md_path, docx_path, title="Technical Report"):
    doc = Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        
        # parse rows
        parsed_rows = []
        for r in table_rows:
            # strip leading/trailing pipe and split
            cleaned = r.strip()
            if cleaned.startswith('|'): cleaned = cleaned[1:]
            if cleaned.endswith('|'): cleaned = cleaned[:-1]
            cols = [c.strip() for c in cleaned.split('|')]
            # ignore separator rows like |---|---|
            if all(re.match(r'^:?-+:?$', c) for c in cols if c):
                continue
            parsed_rows.append(cols)
            
        if not parsed_rows:
            table_rows = []
            in_table = False
            return

        num_cols = max(len(r) for r in parsed_rows)
        tbl = doc.add_table(rows=len(parsed_rows), cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = True
        
        for i, row in enumerate(parsed_rows):
            is_header = (i == 0)
            for j in range(num_cols):
                cell = tbl.cell(i, j)
                text = row[j] if j < len(row) else ""
                # remove bold markdown from table cells for cleaner look or style it
                clean_text = text.replace('**', '').replace('`', '')
                cell.text = clean_text
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                run = p.runs[0] if p.runs else p.add_run()
                run.font.name = 'Calibri'
                run.font.size = Pt(9.5)
                
                if is_header:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    set_cell_background(cell, "1E3A8A") # Navy Blue
                else:
                    if i % 2 == 1:
                        set_cell_background(cell, "F8FAFC") # Soft Slate
                    else:
                        set_cell_background(cell, "FFFFFF")
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_after = Pt(6)
        table_rows = []
        in_table = False

    def flush_code():
        nonlocal in_code_block, code_lines
        if not code_lines:
            in_code_block = False
            return
        code_text = "".join(code_lines)
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.2)
        run = p.add_run(code_text.strip('\n'))
        run.font.name = 'Consolas'
        run.font.size = Pt(9.0)
        run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        code_lines = []
        in_code_block = False

    def add_styled_paragraph(text, style='Normal', bullet=False, space_after=4):
        p = doc.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(space_after)
        if bullet:
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.15)
            run_bullet = p.add_run("• ")
            run_bullet.font.bold = True
            run_bullet.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

        # Parse basic markdown inline formatting (**bold**, `code`, *italic*)
        pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)')
        tokens = pattern.split(text)
        for token in tokens:
            if not token:
                continue
            if token.startswith('**') and token.endswith('**'):
                r = p.add_run(token[2:-2])
                r.font.bold = True
                r.font.name = 'Calibri'
            elif token.startswith('`') and token.endswith('`'):
                r = p.add_run(token[1:-1])
                r.font.name = 'Consolas'
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x8A, 0x1C, 0x14)
            elif token.startswith('*') and token.endswith('*'):
                r = p.add_run(token[1:-1])
                r.font.italic = True
                r.font.name = 'Calibri'
            else:
                r = p.add_run(token)
                r.font.name = 'Calibri'
        return p

    for line in lines:
        stripped = line.rstrip('\r\n')

        # Code blocks
        if stripped.startswith('```'):
            if in_code_block:
                flush_code()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(stripped + '\n')
            continue

        # Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_rows.append(stripped)
            continue
        else:
            if in_table:
                flush_table()

        # Headings
        if stripped.startswith('# '):
            h = doc.add_heading(level=1)
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after = Pt(6)
            run = h.add_run(stripped[2:])
            run.font.name = 'Calibri'
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A) # Deep Slate
            run.font.bold = True
            run.font.size = Pt(20)
        elif stripped.startswith('## '):
            h = doc.add_heading(level=2)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(4)
            run = h.add_run(stripped[3:])
            run.font.name = 'Calibri'
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Deep Blue
            run.font.bold = True
            run.font.size = Pt(15)
        elif stripped.startswith('### '):
            h = doc.add_heading(level=3)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(3)
            run = h.add_run(stripped[4:])
            run.font.name = 'Calibri'
            run.font.color.rgb = RGBColor(0x03, 0x69, 0xA1) # Teal/Sky
            run.font.bold = True
            run.font.size = Pt(12)
        elif stripped.startswith('#### '):
            h = doc.add_heading(level=4)
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
            run = h.add_run(stripped[5:])
            run.font.name = 'Calibri'
            run.font.bold = True
            run.font.size = Pt(11)
        elif stripped.startswith('---'):
            # horizontal divider
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run("―" * 45)
            r.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
        elif stripped.startswith('* ') or stripped.startswith('- '):
            add_styled_paragraph(stripped[2:], bullet=True)
        elif re.match(r'^\d+\.\s', stripped):
            # Numbered item
            prefix_match = re.match(r'^(\d+\.)\s*(.*)', stripped)
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.15)
            r_num = p.add_run(prefix_match.group(1) + " ")
            r_num.font.bold = True
            r_num.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
            # parse the rest
            pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)')
            tokens = pattern.split(prefix_match.group(2))
            for token in tokens:
                if not token: continue
                if token.startswith('**') and token.endswith('**'):
                    r = p.add_run(token[2:-2]); r.font.bold = True; r.font.name = 'Calibri'
                elif token.startswith('`') and token.endswith('`'):
                    r = p.add_run(token[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(9.5)
                else:
                    r = p.add_run(token); r.font.name = 'Calibri'
        elif stripped.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run("❝ " + stripped[2:])
            r.font.italic = True
            r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        elif stripped.strip():
            add_styled_paragraph(stripped)
        else:
            # blank line
            pass

    if in_code_block:
        flush_code()
    if in_table:
        flush_table()

    doc.save(docx_path)
    print(f"Generated DOCX at: {docx_path}")

if __name__ == "__main__":
    base_dir = r"c:\Users\garyl\OneDrive\Desktop\prototype"
    
    # 1. System Reference
    sys_ref_md = os.path.join(base_dir, "SYSTEM_REFERENCE.md")
    sys_ref_docx = os.path.join(base_dir, "SYSTEM_REFERENCE.docx")
    markdown_to_docx(sys_ref_md, sys_ref_docx, title="System Reference & Architecture Handbook")
    
    # 2. Project Report
    rep_md = os.path.join(base_dir, "PROJECT_REPORT.md")
    rep_docx = os.path.join(base_dir, "PROJECT_REPORT.docx")
    markdown_to_docx(rep_md, rep_docx, title="Project Report")

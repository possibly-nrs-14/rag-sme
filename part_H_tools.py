# part_H_tools.py
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from docx import Document
from pptx import Presentation

def export_qa_pdf(answer, sources, out_path):
    import textwrap
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=LETTER)
    w, h = LETTER
    y = h - 72
    c.setFont("Times-Roman", 14)
    for line in (answer or "").split("\n"):
        wrapped_lines = textwrap.wrap(line, width=95)
        for wrapped in wrapped_lines:
            c.drawString(72, y, wrapped)
            y -= 18
            if y < 72:
                c.showPage()
                y = h - 72
    y -= 12
    c.setFont("Times-Bold", 12)
    c.drawString(72, y, "Sources:")
    y -= 18
    c.setFont("Times-Roman", 10)
    for s in sources or []:
        line = f"- {s.get('book')} | chunk={s.get('chunk_id')} | g={s.get('granularity')} | pos={s.get('position')}"
        wrapped_lines = textwrap.wrap(line, width=95)
        for wrapped in wrapped_lines:
            c.drawString(72, y, wrapped)
            y -= 14
            if y < 72:
                c.showPage()
                y = h - 72
    c.save()    

def export_quiz_pdf(items, out_path):
    import textwrap
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=LETTER)
    w, h = LETTER
    y = h - 72
    num = 1
    for it in items:
        q = it.get("question") or ""
        opts = it.get("options") or {}
        c.setFont("Times-Bold", 12)
        
        # Wrap question text
        wrapped_q = textwrap.wrap(f"{num}. {q}", width=95)
        for wrapped in wrapped_q:
            c.drawString(72, y, wrapped)
            y -= 18
            if y < 120:
                c.showPage()
                y = h - 72
        
        c.setFont("Times-Roman", 11)
        for k in ["A","B","C","D"]:
            v = opts.get(k, "")
            wrapped_opts = textwrap.wrap(f"{k}. {v}", width=90)
            for wrapped in wrapped_opts:
                c.drawString(90, y, wrapped)
                y -= 16
                if y < 120:
                    c.showPage()
                    y = h - 72
        y -= 6
        num += 1
    
    c.showPage()
    c.setFont("Times-Bold", 12)
    c.drawString(72, h-72, "Answer Key")
    y = h - 96
    c.setFont("Times-Roman", 11)
    for i, it in enumerate(items, 1):
        c.drawString(72, y, f"{i}. {it.get('correct')}")
        y -= 16
        if y < 72:
            c.showPage()
            y = h - 72
    c.save()

def export_qa_docx(answer, sources, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = Document()
    d.add_heading("Answer", level=1)
    for line in (answer or "").split("\n"):
        d.add_paragraph(line)
    d.add_heading("Sources", level=2)
    for s in sources or []:
        d.add_paragraph(f"{s.get('book')} | chunk={s.get('chunk_id')} | g={s.get('granularity')} | pos={s.get('position')}")
    d.save(out_path)

def export_quiz_docx(items, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = Document()
    d.add_heading("Quiz", level=1)
    for i, it in enumerate(items, 1):
        d.add_paragraph(f"{i}. {it.get('question')}")
        opts = it.get("options") or {}
        for k in ["A","B","C","D"]:
            d.add_paragraph(f"{k}. {opts.get(k,'')}", style="List Bullet")
    d.add_page_break()
    d.add_heading("Answer Key", level=1)
    for i, it in enumerate(items, 1):
        d.add_paragraph(f"{i}. {it.get('correct')}")
    d.save(out_path)

def export_quiz_pptx(items, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prs = Presentation()
    for i, it in enumerate(items, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Q{i}: {it.get('question')}"
        body = slide.shapes.placeholders[1].text_frame
        body.clear()
        opts = it.get("options") or {}
        for k in ["A","B","C","D"]:
            body.add_paragraph().text = f"{k}. {opts.get(k,'')}"
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Answer Key"
    body = slide.shapes.placeholders[1].text_frame
    body.clear()
    for i, it in enumerate(items, 1):
        body.add_paragraph().text = f"{i}. {it.get('correct')}"
    prs.save(out_path)

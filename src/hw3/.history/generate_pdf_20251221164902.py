#!/usr/bin/env python3
"""Convert markdown report to PDF"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re

def create_pdf_report():
    # Read markdown file
    with open('hw3_report_1155249290.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    pdf_filename = 'hw3_report_1155249290.pdf'
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=20,
        textColor=colors.HexColor('#d63031'),
        fontName='Courier'
    )
    
    # Parse markdown content
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Title (# )
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            elements.append(Paragraph(text, title_style))
            elements.append(Spacer(1, 0.1*inch))
        
        # Subtitle (**text**)
        elif line.startswith('**Student ID:**') or line.startswith('**Date:**'):
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line)
            elements.append(Paragraph(text, subtitle_style))
        
        # Section heading (## )
        elif line.startswith('## '):
            text = line[3:].strip()
            elements.append(Spacer(1, 0.15*inch))
            elements.append(Paragraph(text, heading1_style))
        
        # Subsection heading (### )
        elif line.startswith('### '):
            text = line[4:].strip()
            elements.append(Paragraph(text, heading2_style))
        
        # Horizontal rule (---)
        elif line == '---':
            elements.append(Spacer(1, 0.1*inch))
        
        # Table detection
        elif '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
            # Parse table
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            i -= 1
            
            # Convert to table data
            table_data = []
            for tline in table_lines:
                if '---' not in tline:  # Skip separator line
                    cells = [cell.strip() for cell in tline.split('|')[1:-1]]
                    table_data.append(cells)
            
            if table_data:
                # Create table
                t = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                elements.append(t)
                elements.append(Spacer(1, 0.1*inch))
        
        # Bullet points (- or *)
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            # Handle bold
            text = text.replace('**', '<b>').replace('**', '</b>')
            # Handle code
            text = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#d63031">\1</font>', text)
            elements.append(Paragraph('• ' + text, bullet_style))
        
        # Code blocks (```)
        elif line.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code_text = '<br/>'.join(code_lines)
            elements.append(Paragraph(code_text, code_style))
            elements.append(Spacer(1, 0.1*inch))
        
        # Regular paragraph
        elif line and not line.startswith('#'):
            text = line
            # Handle bold
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            # Handle inline code
            text = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#d63031">\1</font>', text)
            # Handle checkmarks
            text = text.replace('✅', '<font color="green">✓</font>')
            text = text.replace('❌', '<font color="red">✗</font>')
            
            elements.append(Paragraph(text, body_style))
        
        # Empty line
        elif not line:
            elements.append(Spacer(1, 0.05*inch))
        
        i += 1
    
    # Build PDF
    doc.build(elements)
    print(f"PDF report generated: {pdf_filename}")
    print(f"File size: {os.path.getsize(pdf_filename) / 1024:.1f} KB")

if __name__ == '__main__':
    import os
    create_pdf_report()

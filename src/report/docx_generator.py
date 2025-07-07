"""
DOCX report generator module with styling and template support.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import re

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import Cm

from src.config import TEMPLATE_PATH


class ReportGenerator:
    """Generates formatted DOCX reports with consistent styling."""
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize report generator with optional template.
        
        Args:
            template_path: Path to DOCX template file (optional)
        """
        self.template_path = template_path or TEMPLATE_PATH
        
        # Initialize document
        if template_path and Path(template_path).exists():
            self.doc = Document(template_path)
        else:
            self.doc = Document()
            self._setup_default_styles()

    def _setup_default_styles(self):
        """Set up default document styles."""
        # Page margins
        sections = self.doc.sections
        for section in sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)
        
        # Title style
        title_style = self.doc.styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.name = 'Calibri'
        title_style.font.size = Pt(28)
        title_style.font.bold = True
        title_style.font.color.rgb = RGBColor(31, 73, 125)  # Dark blue
        title_style.paragraph_format.space_before = Pt(0)
        title_style.paragraph_format.space_after = Pt(20)
        title_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        
        # Heading styles
        for i, size in enumerate([20, 16, 14], 1):
            heading_style = self.doc.styles.add_style(f'CustomHeading{i}', WD_STYLE_TYPE.PARAGRAPH)
            heading_style.font.name = 'Calibri'
            heading_style.font.size = Pt(size)
            heading_style.font.bold = True
            heading_style.font.color.rgb = RGBColor(68, 84, 106)  # Slate gray
            heading_style.paragraph_format.space_before = Pt(20)
            heading_style.paragraph_format.space_after = Pt(12)
            heading_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            heading_style.paragraph_format.keep_with_next = True
        
        # Body style
        body_style = self.doc.styles.add_style('CustomBody', WD_STYLE_TYPE.PARAGRAPH)
        body_style.font.name = 'Calibri'
        body_style.font.size = Pt(11)
        body_style.paragraph_format.space_before = Pt(6)
        body_style.paragraph_format.space_after = Pt(6)
        body_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        
        # List style
        list_style = self.doc.styles.add_style('CustomList', WD_STYLE_TYPE.PARAGRAPH)
        list_style.font.name = 'Calibri'
        list_style.font.size = Pt(11)
        list_style.paragraph_format.space_before = Pt(3)
        list_style.paragraph_format.space_after = Pt(3)
        list_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        list_style.paragraph_format.left_indent = Inches(0.25)
        
        # Quote style
        quote_style = self.doc.styles.add_style('CustomQuote', WD_STYLE_TYPE.PARAGRAPH)
        quote_style.font.name = 'Calibri'
        quote_style.font.italic = True
        quote_style.font.size = Pt(11)
        quote_style.font.color.rgb = RGBColor(89, 89, 89)  # Dark gray
        quote_style.paragraph_format.left_indent = Inches(0.5)
        quote_style.paragraph_format.space_before = Pt(12)
        quote_style.paragraph_format.space_after = Pt(12)
        quote_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove headers
        text = re.sub(r'#+\s+', '', text)
        # Remove bold/italic
        text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Remove links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove blockquotes
        text = re.sub(r'>\s+', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        return text

    def add_title_page(self, title: str, subtitle: Optional[str] = None,
                      logo_path: Optional[str] = None):
        """
        Add a formatted title page to the document.
        
        Args:
            title: Report title
            subtitle: Optional subtitle
            logo_path: Optional path to logo image
        """
        # Add logo if provided
        if logo_path and Path(logo_path).exists():
            self.doc.add_picture(logo_path, width=Inches(2.5))
            self.doc.add_paragraph()  # Spacing
        
        # Add title
        title_para = self.doc.add_paragraph(title, style='CustomTitle')
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add subtitle if provided
        if subtitle:
            subtitle_para = self.doc.add_paragraph(subtitle, style='CustomHeading2')
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add date
        date_para = self.doc.add_paragraph(
            datetime.datetime.now().strftime("%B %d, %Y"),
            style='CustomBody'
        )
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add page break
        self.doc.add_page_break()

    def add_section(self, title: str, content: str, level: int = 1):
        """
        Add a new section with heading and content.
        
        Args:
            title: Section title
            content: Section content text
            level: Heading level (1-3)
        """
        # Add heading
        self.doc.add_paragraph(title, style=f'CustomHeading{level}')
        
        # Clean and process content
        content = self._clean_markdown(content)
        
        # Split into paragraphs and add
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            # Check if it's a list item
            if para.strip().startswith('•') or para.strip().startswith('-'):
                p = self.doc.add_paragraph(style='CustomList')
                p.add_run(para.strip()[2:].strip())
            else:
                self.doc.add_paragraph(para, style='CustomBody')

    def add_insights(self, insights: Dict[str, Any]):
        """
        Add an insights section with key findings.
        
        Args:
            insights: Dictionary of insight categories and their content
        """
        self.doc.add_paragraph("Key Insights", style='CustomHeading1')
        
        for category, items in insights.items():
            # Add category heading
            self.doc.add_paragraph(category, style='CustomHeading2')
            
            # Add items as bullet points
            for item in items:
                p = self.doc.add_paragraph(style='CustomList')
                p.add_run("• " + self._clean_markdown(item))

    def add_metrics_table(self, metrics: Dict[str, Any]):
        """
        Add a table of metrics or KPIs.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Add heading
        self.doc.add_paragraph("Key Metrics", style='CustomHeading2')
        
        # Create table
        table = self.doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        table.autofit = True
        
        # Add header row
        header_cells = table.rows[0].cells
        header_cells[0].text = "Metric"
        header_cells[1].text = "Value"
        
        # Style header
        for cell in header_cells:
            cell.paragraphs[0].style = self.doc.styles['CustomHeading3']
        
        # Add metric rows
        for metric, value in metrics.items():
            row_cells = table.add_row().cells
            row_cells[0].text = metric
            row_cells[1].text = str(value)
            
            # Style cells
            for cell in row_cells:
                cell.paragraphs[0].style = self.doc.styles['CustomBody']

    def add_quotes(self, quotes: list):
        """
        Add a section of highlighted quotes.
        
        Args:
            quotes: List of quote strings
        """
        for quote in quotes:
            quote = self._clean_markdown(quote)
            p = self.doc.add_paragraph(style='CustomQuote')
            p.add_run(f'"{quote}"')

    def save(self, output_path: str):
        """
        Save the document to disk.
        
        Args:
            output_path: Path where to save the DOCX file
        """
        self.doc.save(output_path) 
"""
DOCX report generator module with Norstella styling and one-page limit.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import datetime
import re
import io
import logging

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Twips
from docx.shared import Length

import matplotlib.pyplot as plt
import numpy as np

from src.config import TEMPLATE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartType:
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    STACKED_BAR = "stacked_bar"
    HORIZONTAL_BAR = "horizontal_bar"

class ReportGenerator:
    """Generates formatted one-page DOCX reports with Norstella styling."""
    
    NORSTELLA_BLUE = RGBColor(31, 73, 125)  # Primary brand color
    NORSTELLA_GRAY = RGBColor(68, 84, 106)  # Secondary color
    
    # Page size constants (in inches)
    PAGE_WIDTH = 8.5
    PAGE_HEIGHT = 11.0
    MARGIN = 1.0
    CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)
    CONTENT_HEIGHT = PAGE_HEIGHT - (2 * MARGIN)
    
    def __init__(self, template_path: Optional[str] = None):
        """Initialize report generator."""
        self.doc = Document()
        self._setup_norstella_styles()
        self._setup_page_layout()
        self.content_height = 0  # Track content height

    def _setup_page_layout(self):
        """Set up page layout for one-page format."""
        section = self.doc.sections[0]
        section.page_height = Inches(self.PAGE_HEIGHT)
        section.page_width = Inches(self.PAGE_WIDTH)
        section.left_margin = Inches(self.MARGIN)
        section.right_margin = Inches(self.MARGIN)
        section.top_margin = Inches(self.MARGIN)
        section.bottom_margin = Inches(self.MARGIN)

    def _setup_norstella_styles(self):
        """Set up document styles for one-page format."""
        # Title style (smaller than before)
        title_style = self.doc.styles.add_style('NorstTitle', WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.name = 'Segoe UI'
        title_style.font.size = Pt(18)  # Reduced from 24
        title_style.font.bold = True
        title_style.font.color.rgb = self.NORSTELLA_BLUE
        title_style.paragraph_format.space_before = Pt(12)
        title_style.paragraph_format.space_after = Pt(6)
        title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Subtitle style (smaller)
        subtitle_style = self.doc.styles.add_style('NorstSubtitle', WD_STYLE_TYPE.PARAGRAPH)
        subtitle_style.font.name = 'Segoe UI'
        subtitle_style.font.size = Pt(12)  # Reduced from 16
        subtitle_style.font.color.rgb = self.NORSTELLA_GRAY
        subtitle_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_style.paragraph_format.space_after = Pt(12)
        
        # Section header style (smaller)
        header_style = self.doc.styles.add_style('NorstHeader', WD_STYLE_TYPE.PARAGRAPH)
        header_style.font.name = 'Segoe UI'
        header_style.font.size = Pt(11)  # Reduced from 14
        header_style.font.bold = True
        header_style.font.color.rgb = self.NORSTELLA_BLUE
        header_style.paragraph_format.space_before = Pt(6)
        header_style.paragraph_format.space_after = Pt(3)
        
        # Body style (compact)
        body_style = self.doc.styles.add_style('NorstBody', WD_STYLE_TYPE.PARAGRAPH)
        body_style.font.name = 'Calibri'
        body_style.font.size = Pt(10)  # Reduced from 11
        body_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        body_style.paragraph_format.line_spacing = 1.0  # Single spacing
        body_style.paragraph_format.space_after = Pt(3)
        
        # List style (compact)
        list_style = self.doc.styles.add_style('NorstList', WD_STYLE_TYPE.PARAGRAPH)
        list_style.font.name = 'Calibri'
        list_style.font.size = Pt(10)
        list_style.paragraph_format.line_spacing = 1.0
        list_style.paragraph_format.left_indent = Inches(0.15)
        list_style.paragraph_format.space_after = Pt(2)

    def add_section(self, title: str, content: str, max_length: Optional[int] = None):
        """
        Add a section with length control.
        
        Args:
            title: Section title
            content: Section content text
            max_length: Maximum number of characters (optional)
        """
        # Add section header
        header = self.doc.add_paragraph(title, style='NorstHeader')
        
        # Truncate content if needed
        if max_length and len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        # Process and add content
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip().startswith(('•', '-', '*')):
                # Handle bullet points
                p = self.doc.add_paragraph(style='NorstList')
                p.add_run('• ' + para.strip()[1:].strip())
            else:
                # Regular paragraph
                self.doc.add_paragraph(para, style='NorstBody')

    def add_cover_page(self, title: str, subtitle: Optional[str] = None,
                      logo_path: Optional[str] = None):
        """Add a compact cover section."""
        if logo_path and Path(logo_path).exists():
            self.doc.add_picture(logo_path, width=Inches(1.5))  # Smaller logo
        
        title_para = self.doc.add_paragraph(title, style='NorstTitle')
        
        if subtitle:
            subtitle_para = self.doc.add_paragraph(subtitle, style='NorstSubtitle')

    def add_chart(self, chart_data: Dict[str, Any], chart_type: str = ChartType.BAR,
                 title: Optional[str] = None, caption: Optional[str] = None,
                 width: float = 6, height: float = 4, new_page: bool = False):
        """
        Add a chart to the document.
        
        Args:
            chart_data: Dictionary containing chart data and configuration
            chart_type: Type of chart to create (bar, line, pie, etc.)
            title: Chart title
            caption: Chart caption or description
            width: Chart width in inches
            height: Chart height in inches
            new_page: Whether to place chart on a new page
        """
        if new_page:
            self.doc.add_page_break()
        
        # Create figure with Norstella styling
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Set Norstella colors
        colors = ['#1F497D', '#445A6A', '#0070C0', '#595959']
        
        # Create chart based on type
        if chart_type == ChartType.BAR:
            ax.bar(chart_data['x'], chart_data['y'], color=colors[0])
        elif chart_type == ChartType.LINE:
            ax.plot(chart_data['x'], chart_data['y'], color=colors[0], linewidth=2, marker='o')
        elif chart_type == ChartType.PIE:
            ax.pie(chart_data['values'], labels=chart_data['labels'], colors=colors,
                  autopct='%1.1f%%', startangle=90)
        elif chart_type == ChartType.STACKED_BAR:
            bottom = np.zeros(len(chart_data['x']))
            for i, y in enumerate(chart_data['y_series']):
                ax.bar(chart_data['x'], y, bottom=bottom, color=colors[i % len(colors)],
                      label=chart_data['series_labels'][i])
                bottom += y
            ax.legend()
        elif chart_type == ChartType.HORIZONTAL_BAR:
            ax.barh(chart_data['y'], chart_data['x'], color=colors[0])
        
        # Style the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if chart_type != ChartType.PIE:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save chart to memory
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
        img_stream.seek(0)
        plt.close()
        
        # Add title if provided
        if title:
            self.doc.add_paragraph(title, style='NorstChartTitle')
        
        # Add chart to document
        self.doc.add_picture(img_stream, width=Inches(width))
        
        # Add caption if provided
        if caption:
            self.doc.add_paragraph(caption, style='NorstChartCaption')
        
        if new_page:
            self.doc.add_page_break()

    def add_chart_from_file(self, image_path: str, title: Optional[str] = None,
                          caption: Optional[str] = None, width: float = 6.0):
        """
        Add a chart from an image file to the document.
        
        Args:
            image_path: Path to the image file
            title: Optional title to display above the chart
            caption: Optional caption to display below the chart
            width: Width of the chart in inches
        """
        if title:
            title_para = self.doc.add_paragraph(title, style='NorstChartTitle')
        
        # Add the image
        self.doc.add_picture(image_path, width=Inches(width))
        
        if caption:
            caption_para = self.doc.add_paragraph(caption, style='NorstChartCaption')
        
        # Add some space after the chart
        self.doc.add_paragraph()

    def _clean_markdown(self, text: str) -> str:
        """Process markdown-style formatting in text."""
        # Remove headers
        text = re.sub(r'#+\s+', '', text)
        
        # Handle bold text
        text = re.sub(r'\*\*(.*?)\*\*', lambda m: self._make_bold(m.group(1)), text)
        
        # Handle italic text
        text = re.sub(r'\*(.*?)\*', lambda m: self._make_italic(m.group(1)), text)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        return text

    def _make_bold(self, text: str) -> str:
        """Mark text for bold formatting."""
        return f'<b>{text}</b>'

    def _make_italic(self, text: str) -> str:
        """Mark text for italic formatting."""
        return f'<i>{text}</i>'

    def save(self, filename: str):
        """
        Save the document with page limit check.
        
        Args:
            filename: Output filename
        """
        # Count pages before saving
        self.doc.save(filename)
        doc = Document(filename)
        
        if len(doc.sections) > 1:
            # Document is too long, need to truncate
            logging.warning("Document exceeds one page, truncating content...")
            self.doc = Document()
            self._setup_norstella_styles()
            self._setup_page_layout()
            
            # Regenerate with shorter content
            self.add_cover_page(
                "Flash Report",
                f"Generated on {datetime.datetime.now().strftime('%B %d, %Y')}"
            )
            
            # Add sections with reduced content
            for section in doc.sections[0].text.split('\n\n'):
                if section.strip():
                    title = section.split('\n')[0]
                    content = '\n'.join(section.split('\n')[1:])
                    self.add_section(title, content, max_length=300)  # Limit each section
            
            # Save truncated version
            self.doc.save(filename) 
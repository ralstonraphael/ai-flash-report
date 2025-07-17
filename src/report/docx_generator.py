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

from src.config import TEMPLATE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def _parse_markdown_text(self, text: str, paragraph):
        """
        Parse markdown-formatted text and add it to a paragraph with proper Word formatting.
        
        Args:
            text: Text containing markdown formatting
            paragraph: docx paragraph object to add formatted text to
        """
        # Split text by markdown patterns while preserving the delimiters
        # This regex captures bold (**text**), italic (*text*), and regular text
        pattern = r'(\*\*.*?\*\*|\*.*?\*|[^*]+)'
        parts = re.findall(pattern, text)
        
        for part in parts:
            if not part.strip():
                continue
                
            if part.startswith('**') and part.endswith('**'):
                # Bold text
                bold_text = part[2:-2]  # Remove ** markers
                run = paragraph.add_run(bold_text)
                run.bold = True
            elif part.startswith('*') and part.endswith('*') and not part.startswith('**'):
                # Italic text  
                italic_text = part[1:-1]  # Remove * markers
                run = paragraph.add_run(italic_text)
                run.italic = True
            else:
                # Regular text
                paragraph.add_run(part)

    def _add_formatted_paragraph(self, text: str, style: str = 'NorstBody'):
        """
        Add a paragraph with markdown formatting converted to Word formatting.
        
        Args:
            text: Text that may contain markdown formatting
            style: Paragraph style to apply
        """
        # Clean up the text first
        text = text.strip()
        if not text:
            return
        
        # Create the paragraph
        paragraph = self.doc.add_paragraph(style=style)
        
        # Parse and add the formatted text
        self._parse_markdown_text(text, paragraph)

    def _process_section_content(self, content: str) -> List[str]:
        """
        Process section content and split into paragraphs, handling markdown headers.
        
        Args:
            content: Raw content text with possible markdown formatting
            
        Returns:
            List of processed paragraphs with (type, text) tuples
        """
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        processed_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if this is a markdown header (starts with **)
            if para.startswith('**') and '**' in para[2:]:
                # This is a subheader - extract the header text
                header_match = re.match(r'\*\*(.*?)\*\*(.*)', para, re.DOTALL)
                if header_match:
                    header_text = header_match.group(1).strip()
                    remaining_text = header_match.group(2).strip()
                    
                    # Only add header if there's meaningful content following it
                    if header_text and remaining_text and len(remaining_text) > 20:
                        processed_paragraphs.append(('header', header_text))
                        processed_paragraphs.append(('body', remaining_text))
                    elif header_text and remaining_text:
                        # If there's a header but minimal content, combine them
                        combined_text = f"{header_text}: {remaining_text}"
                        processed_paragraphs.append(('body', combined_text))
                    elif remaining_text:
                        # If there's only content, treat as regular paragraph
                        processed_paragraphs.append(('body', remaining_text))
                else:
                    # Fallback - treat as regular text
                    processed_paragraphs.append(('body', para))
            else:
                # Regular paragraph - only add if it has substantial content
                if len(para) > 10:  # Minimum length threshold
                    processed_paragraphs.append(('body', para))
        
        return processed_paragraphs

    def add_section(self, title: str, content: str, max_length: Optional[int] = None):
        """
        Add a section with proper Word formatting instead of markdown.
        
        Args:
            title: Section title
            content: Section content text (may contain markdown)
            max_length: Maximum number of characters (optional)
        """
        # Skip sections with no meaningful content
        if not content or len(content.strip()) < 20:
            return
        
        # Add section header
        header = self.doc.add_paragraph(title, style='NorstHeader')
        
        # Truncate content if needed
        if max_length and len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        # Process the content to handle markdown formatting
        processed_content = self._process_section_content(content)
        
        # Only add section if there's actual content to display
        if not processed_content:
            return
        
        for content_type, text in processed_content:
            if content_type == 'header':
                # Add as a sub-header with bold formatting
                sub_header = self.doc.add_paragraph(style='NorstBody')
                run = sub_header.add_run(text)
                run.bold = True
                run.font.size = Pt(10)
            elif content_type == 'body':
                # Check if it's a bullet point
                if text.strip().startswith(('•', '-', '*')):
                    # Handle bullet points
                    bullet_text = text.strip()[1:].strip()
                    if bullet_text:  # Only add if there's actual content
                        p = self.doc.add_paragraph(style='NorstList')
                        p.add_run(bullet_text)
                else:
                    # Regular paragraph
                    self._add_formatted_paragraph(text)

    def add_cover_page(self, title: str, subtitle: Optional[str] = None):
        """Add a cover page to the document."""
        # Add title
        title_para = self.doc.add_paragraph(title, style='NorstTitle')
        
        # Add subtitle if provided
        if subtitle:
            subtitle_para = self.doc.add_paragraph(subtitle, style='NorstSubtitle')

    def save(self, filename: str):
        """Save the document to a file."""
        self.doc.save(filename)

    def add_formatted_text(self, text: str, style: str = 'NorstBody'):
        """Add formatted text to the document."""
        self._add_formatted_paragraph(text, style) 
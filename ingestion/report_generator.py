# import re
# from docx import Document
# from docx.shared import Pt
# from docx.enum.text import WD_ALIGN_PARAGRAPH
#
# # No need for WD_SECTION or OxmlElement, ns for basic report generation in this version
#
# def create_flash_report(data: dict, output_filename="Norstella_Flash_Report.docx"):
#     """
#     Generates a Flash Report in DOCX format based on the provided data.
#
#     Args:
#         data (dict): A dictionary containing the extracted information.
#                      Expected keys:
#                      - "executive_summary": str
#                      - "key_takeaways": list of str or single str
#                      - "financial_highlights": list of str or single str
#                      - "report_title": str (e.g., "Q4-24 Norstella Quarterly Market Participant Update")
#                      - "report_date": str (e.g., "February 19th, 2025")
#         output_filename (str): The name of the output DOCX file.
#     """
#     document = Document()
#
#     # --- Header (Simple text for now) ---
#     section = document.sections[0]
#     header = section.header
#     # Ensure there's at least one paragraph in the header
#     if not header.paragraphs:
#         header.add_paragraph()
#     paragraph = header.paragraphs[0]
#     header_run = paragraph.add_run("Norstella")
#     header_run.font.size = Pt(12)
#     header_run.bold = True
#     paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
#
#     # --- Main Content ---
#
#     # Add Report Title
#     title_paragraph = document.add_paragraph()
#     title_run = title_paragraph.add_run(data.get("report_title", "Quarterly Market Participant Update"))
#     title_run.font.size = Pt(24)
#     title_run.bold = True
#     title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
#     document.add_paragraph(f"Report Date: {data.get('report_date', 'Date Not Available')}", style='Intense Quote')
#
#
#     # Add Executive Summary
#     document.add_heading("EXECUTIVE SUMMARY", level=1)
#     executive_summary_text = data.get("executive_summary", "Executive summary not available.")
#     # Remove common boilerplate if present
#     executive_summary_text = re.sub(r'### Strategic Summary.*?(Company Background:.*?)?|\*\*Company Background:\*\*', '', executive_summary_text, flags=re.DOTALL | re.IGNORECASE).strip()
#     executive_summary_text = re.sub(r'\*\*Products:.*|\*\*Key Insights:.*', '', executive_summary_text, flags=re.DOTALL | re.IGNORECASE).strip()
#
#     document.add_paragraph(executive_summary_text)
#
#     # Helper function to add bullet points, cleaning up LLM boilerplate
#     def add_bullet_points(doc, heading_text, content_key):
#         doc.add_heading(heading_text, level=1)
#         content = data.get(content_key)
#         if content:
#             if isinstance(content, str):
#                 # Remove common boilerplate from list items
#                 content = re.sub(r'### Strategic Summary.*?(Company Background:.*?)?|\*\*Company Background:\*\*', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
#                 content = re.sub(r'\*\*Products:.*|\*\*Key Insights:.*', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
#
#                 # Split by newlines, handling numbered lists or bullet points
#                 points = [p.strip() for p in content.split('\n') if p.strip()]
#                 # Refine splitting for numbered lists (e.g., "1. Point") or leading bullets
#                 cleaned_points = []
#                 for point in points:
#                     # Remove leading numbers/bullets like "1. ", "- ", "* "
#                     point = re.sub(r'^\s*[\d\*\-]+\s*', '', point).strip()
#                     if point: # Only add if not empty after stripping
#                         cleaned_points.append(point)
#
#                 if not cleaned_points: # Fallback if splitting didn't yield points
#                     doc.add_paragraph(content)
#                 else:
#                     for point in cleaned_points:
#                         doc.add_paragraph(point, style='List Bullet')
#             elif isinstance(content, list):
#                 for item in content:
#                     doc.add_paragraph(item, style='List Bullet')
#         else:
#             doc.add_paragraph(f"{heading_text.lower().replace('highlights', 'information').replace('key ', '')} not available.")
#
#     add_bullet_points(document, "KEY TAKEAWAYS", "key_takeaways")
#     add_bullet_points(document, "FINANCIAL HIGHLIGHTS", "financial_highlights")
#
#     # --- Footer ---
#     section = document.sections[0]
#     footer = section.footer
#     # Ensure there's at least one paragraph in the footer
#     if not footer.paragraphs:
#         footer.add_paragraph()
#     footer.paragraphs[0].text = "Confidential - Norstella Internal Report"
#     footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
#
#     document.save(output_filename)
#     print(f"Report '{output_filename}' generated successfully.")
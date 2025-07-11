"""
Streamlit web interface for the Flash Report Generator.
"""
import streamlit as st
from pathlib import Path
import tempfile
import os
import logging
import warnings
import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable specific loggers
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Filter out warnings more comprehensively
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*capture().*")
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*CT_Style.*")

# Set environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMADB_TELEMETRY"] = "False"

# Configure event loop policy for macOS
import sys
if sys.platform == "darwin":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        logger.warning(f"Could not set event loop policy: {e}")

from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.store import VectorStore
from src.llm.query_engine import QueryEngine, QueryIntent
from src.report.docx_generator import ReportGenerator
from src.config import VECTORSTORE_PATH

def init_session_state():
    """Initialize session state variables."""
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = os.getenv("OPENAI_API_KEY")
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload"
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None


def setup_page():
    """Configure page settings and layout."""
    st.set_page_config(
        page_title="Flash Report Generator",
        page_icon="📊",
        layout="wide"
    )
    
    # Create a header with logo and title
    header_col1, header_col2 = st.columns([1, 4])
    
    with header_col1:
        st.image(
            "templates/Images/Norstella_color_positive_RGB_(2).png",
            width=150
        )
    
    with header_col2:
        st.title("Flash Report Generator")
        st.markdown(
            """
            <style>
            .main-header {
                font-family: 'Calibri', sans-serif;
                color: rgb(31, 73, 125);
                padding-top: 0;
                margin-top: -1em;
            }
            .stButton > button {
                background-color: rgb(31, 73, 125);
                color: white;
            }
            .stButton > button:hover {
                background-color: rgb(41, 83, 135);
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


def check_api_key():
    """Check and handle OpenAI API key."""
    if not st.session_state.openai_key:
        with st.sidebar:
            st.subheader("⚙️ Settings")
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.openai_key = api_key
                st.success("API key set!")
                return True
            else:
                st.warning("Please enter your OpenAI API key to continue")
                return False
    return True


def upload_section():
    """Handle document uploads."""
    st.subheader("📁 Upload Documents")
    
    # Debug information
    st.sidebar.write("Debug Information:")
    st.sidebar.write(f"Working Directory: {os.getcwd()}")
    st.sidebar.write(f"Vectorstore Path: {VECTORSTORE_PATH}")
    st.sidebar.write(f"Python Version: {sys.version}")
    
    # Simple file uploader with explicit accept_multiple_files
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, or CSV)",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        key="doc_uploader"
    )
    
    if uploaded_files:
        # Display file information
        st.write("Files received:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.type}, {file.size} bytes)")
        
        # Process button
        if st.button("Process Documents", type="primary", key="process_btn"):
            try:
                # Create necessary directories
                Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
                temp_dir = Path(tempfile.mkdtemp())
                st.write(f"Created temp directory: {temp_dir}")
                
                # Process each file
                with st.status("Processing documents...") as status:
                    processed_files = []
                    
                    for file in uploaded_files:
                        try:
                            # Save file
                            temp_path = temp_dir / file.name
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                            processed_files.append(temp_path)
                            status.write(f"✓ Saved {file.name}")
                            
                        except Exception as e:
                            st.error(f"Error saving {file.name}: {str(e)}")
                            continue
                    
                    if not processed_files:
                        st.error("No files were successfully processed.")
                        return
                    
                    try:
                        # Load documents
                        status.write("Loading documents...")
                        loader = DocumentLoader()
                        documents = loader.load_batch(processed_files)
                        status.write(f"✓ Loaded {len(documents)} document chunks")
                        
                        # Create vector store
                        status.write("Creating knowledge base...")
                        vectorstore = VectorStore()
                        collection_name = f"docs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        vectorstore.create_collection(documents, collection_name)
                        
                        # Update session state
                        st.session_state.documents_loaded = True
                        st.session_state.vectorstore = vectorstore
                        st.session_state.collection_name = collection_name
                        
                        # Success message
                        st.success(f"""
                        ✅ Successfully processed {len(processed_files)} files:
                        - Created {len(documents)} text chunks
                        - Collection name: {collection_name}
                        """)
                        
                        # Guide to next step
                        st.info("👉 Click the Analysis tab above to start exploring your documents")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.error("Please try again with different files or contact support.")
                        logger.error(f"Document processing error: {str(e)}")
                
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error in upload section: {str(e)}")
            
            finally:
                # Cleanup
                try:
                    for path in processed_files:
                        path.unlink(missing_ok=True)
                    temp_dir.rmdir()
                except Exception as e:
                    logger.error(f"Cleanup error: {str(e)}")
    
    else:
        # Help text when no files are uploaded
        st.info("""
        📋 Instructions:
        1. Click 'Browse files' above or drag and drop your documents
        2. Select one or more files (PDF, DOCX, or CSV)
        3. Click 'Process Documents' to analyze them
        """)
        
        # Example file formats
        st.markdown("""
        Supported file formats:
        - PDF (`.pdf`): Reports, articles, documents
        - Word (`.docx`): Microsoft Word documents
        - CSV (`.csv`): Spreadsheet data
        """)


def analysis_section():
    """Handle document analysis and querying."""
    if not st.session_state.documents_loaded:
        st.info("⚠️ Please upload and process documents first")
        return
    
    st.subheader("🔍 Document Analysis")
    
    # Recommended prompts based on content
    with st.expander("💡 Recommended Prompts", expanded=True):
        st.markdown("""
        Based on your uploaded documents, here are some recommended prompts:
        
        **Quick Analysis:**
        - "Generate an executive summary of the key points"
        - "What are the most significant changes or updates?"
        - "What are the key strategic implications?"
        
        **Detailed Analysis:**
        - "Analyze the company's market position and competitive landscape"
        - "Extract key financial metrics and trends"
        - "What are the main product/service updates?"
        
        **Strategic Insights:**
        - "What are the key strategic recommendations?"
        - "Generate a SWOT analysis"
        - "What are the main opportunities and risks?"
        
        Click any prompt to use it.
        """)
        
        # Quick prompt buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Extract Key Metrics"):
                st.session_state.query = "Extract and analyze key financial and operational metrics"
            if st.button("🎯 Strategic Analysis"):
                st.session_state.query = "Provide a detailed strategic analysis and recommendations"
        with col2:
            if st.button("📈 Market Position"):
                st.session_state.query = "Analyze market position and competitive landscape"
            if st.button("📋 Executive Summary"):
                st.session_state.query = "Generate a comprehensive executive summary"
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "What would you like to know?",
            value=st.session_state.get('query', ''),
            help="Enter your question about the documents",
            height=100
        )
    
    with col2:
        query_type = st.selectbox(
            "Analysis Type",
            ["Summary", "Specific Question", "Data Extraction"],
            help="""
            - Summary: Get a high-level overview
            - Specific Question: Get precise answers
            - Data Extraction: Pull out specific data points
            """
        )
        
        analyze_button = st.button("Analyze", type="primary")
    
    if query and analyze_button:
        try:
            progress_placeholder = st.empty()
            
            def update_progress(message: str):
                progress_placeholder.info(message)
            
            # Map query type to intent
            intent_map = {
                "Summary": QueryIntent.SUMMARY,
                "Specific Question": QueryIntent.SPECIFIC_QUESTION,
                "Data Extraction": QueryIntent.DATA_EXTRACTION
            }
            
            # Get context
            update_progress("Finding relevant information...")
            vs = st.session_state.vectorstore
            context = vs.query_collection(query)
            
            # Generate response with progress tracking
            engine = QueryEngine(timeout=45)  # Increased timeout for complex queries
            try:
                response = engine.generate_response(
                    query=query,
                    context=context,
                    intent=intent_map[query_type],
                    progress_callback=update_progress
                )
                
                # Store the response
                st.session_state.analysis_result = response
                
                # Clear progress message
                progress_placeholder.empty()
                
                # Show results
                st.markdown("### Analysis Results")
                st.write(response)
                
                # Show metrics with progress tracking
                st.markdown("### Response Quality Assessment")
                metrics = engine.evaluate_response(
                    query,
                    response,
                    context,
                    progress_callback=update_progress
                )
                
                # Display metrics in an organized way
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Relevance", f"{metrics['relevance']['score']:.2f}")
                    st.metric("Accuracy", f"{metrics['accuracy']['score']:.2f}")
                    st.metric("Completeness", f"{metrics['completeness']['score']:.2f}")
                
                with col2:
                    st.metric("Coherence", f"{metrics['coherence']['score']:.2f}")
                    st.metric("Conciseness", f"{metrics['conciseness']['score']:.2f}")
                
            except QueryTimeoutError:
                st.error("Analysis took too long to complete. Please try a more specific query or break it into smaller parts.")
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            st.error("Error analyzing documents. Please try again with a different query.")


def report_section():
    """Handle report generation."""
    if not st.session_state.documents_loaded:
        st.info("⚠️ Please upload and process documents first")
        return
    
    st.subheader("📊 Flash Report Generation")
    
    # Report configuration
    st.markdown("### Report Configuration")
    st.info("📄 Reports now generate comprehensive, full-length content with detailed analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Report Title",
            value="Flash Report",
            help="Enter the title for your report"
        )
    
    with col2:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        report_date = st.text_input(
            "Report Date",
            value=current_date,
            help="Enter the date for your report"
        )
    
    # Section configuration
    st.markdown("### Report Sections")
    st.caption("Select sections to include (each section will provide comprehensive analysis)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_exec_summary = st.checkbox("🧾 Executive Summary", value=True,
            help="Comprehensive overview with detailed analysis (4-5 paragraphs)")
        include_company = st.checkbox("🧭 Company Overview", value=True,
            help="Detailed company analysis including business model and strategy (4-5 paragraphs)")
        include_offerings = st.checkbox("📦 Core Offerings", value=True,
            help="Thorough analysis of products and services (4-5 paragraphs)")
    
    with col2:
        include_market = st.checkbox("📈 Market Position", value=True,
            help="Detailed competitive analysis and market positioning (4-5 paragraphs)")
        include_insights = st.checkbox("🧠 Key Strategic Insights", value=True,
            help="Comprehensive strategic recommendations and analysis (4-5 paragraphs)")
    
    # Generate report button
    if st.button("Generate Report", type="primary"):
        try:
            with st.status("Generating report...") as status:
                # Initialize report generator
                generator = ReportGenerator()
                
                # Add cover page
                status.write("Creating cover page...")
                generator.add_cover_page(
                    title=report_title,
                    subtitle=f"Generated on {report_date}",
                    logo_path="templates/Images/Norstella_color_positive_RGB_(2).png"
                )
                
                # Generate each section
                vs = st.session_state.vectorstore
                engine = QueryEngine()
                
                if include_exec_summary:
                    status.write("Generating executive summary...")
                    context = vs.query_collection(
                        "Generate a comprehensive executive summary highlighting key changes, updates, and why they matter"
                    )
                    content = engine.generate_section_content("executive_summary", context)
                    generator.add_section("Executive Summary", content)
                
                if include_company:
                    status.write("Analyzing company overview...")
                    context = vs.query_collection(
                        "Extract company information, business model, and recent strategic moves"
                    )
                    content = engine.generate_section_content("company_overview", context)
                    generator.add_section("Company Overview", content)
                
                if include_offerings:
                    status.write("Analyzing core offerings...")
                    context = vs.query_collection(
                        "Extract information about products, services, platforms, and recent launches or changes"
                    )
                    content = engine.generate_section_content("core_offerings", context)
                    generator.add_section("Core Offerings", content)
                
                if include_market:
                    status.write("Analyzing market position...")
                    context = vs.query_collection(
                        "Analyze market position, competitors, differentiators, and market trends"
                    )
                    content = engine.generate_section_content("market_position", context)
                    generator.add_section("Market Position", content)
                
                if include_insights:
                    status.write("Generating strategic insights...")
                    context = vs.query_collection(
                        "Generate strategic insights, implications, and recommendations"
                    )
                    content = engine.generate_section_content("strategic_insights", context)
                    generator.add_section("Strategic Insights", content)
                
                # Save report
                status.write("Saving report...")
                report_filename = f"flash_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                generator.save(report_filename)
                
                status.update(label="✅ Report generated successfully!", state="complete")
            
            # Provide download link
            with open(report_filename, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name=report_filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            # Cleanup
            os.remove(report_filename)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            st.error("Error generating report. Please try again.")


def main():
    """Main application function."""
    init_session_state()
    setup_page()
    
    if not check_api_key():
        return
    
    # Navigation - removed Visualization tab
    tab1, tab2, tab3 = st.tabs(["Upload", "Analysis", "Report"])
    
    with tab1:
        upload_section()
    
    with tab2:
        analysis_section()
    
    with tab3:
        report_section()


if __name__ == "__main__":
    main() 
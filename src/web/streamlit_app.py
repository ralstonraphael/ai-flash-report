"""
Streamlit web interface for the Flash Report Generator.
"""
import sys
import os
from pathlib import Path
import logging

# Import torch isolation fix FIRST before any other imports
import os
import sys

# Comprehensive torch isolation
os.environ.update({
    "PYTHONWARNINGS": "ignore",
    "TORCH_LOGS": "ERROR",
    "CUDA_LAUNCH_BLOCKING": "0",
    "TORCH_USE_CUDA_DSA": "0",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none"
})

# Apply torch watcher fix before any streamlit imports
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from fix_torch_watcher import apply_comprehensive_fix
    apply_comprehensive_fix()
except Exception as e:
    print(f"Warning: Could not apply torch fixes: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# CRITICAL: Import torch fix FIRST before any other imports
try:
    import fix_torch_watcher
    logger.info("✅ Torch watcher fix applied successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import torch fix: {e}")
except Exception as e:
    logger.warning(f"⚠️ Torch fix failed: {e}")

import streamlit as st
import tempfile
import warnings
import datetime
import re

# Configure Streamlit for larger file uploads
st.set_page_config(
    page_title="Norstella AI Flash Reports",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Norstella Color Scheme and Custom CSS
st.markdown("""
<style>
    /* Norstella Brand Colors */
    :root {
        --norstella-teal: #00A3A3;
        --norstella-dark-teal: #008080;
        --norstella-light-teal: #B3E5E5;
        --norstella-navy: #1F3A93;
        --norstella-gray: #4A5568;
        --norstella-light-gray: #F7FAFC;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--norstella-light-gray) 0%, #ffffff 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--norstella-teal) 0%, var(--norstella-dark-teal) 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 15px 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 163, 163, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--norstella-light-gray);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--norstella-teal) 0%, var(--norstella-dark-teal) 100%);
        color: white !important;
        border: 2px solid var(--norstella-dark-teal);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--norstella-teal) 0%, var(--norstella-dark-teal) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 163, 163, 0.4);
    }
    
    /* Info boxes */
    .stInfo {
        background-color: var(--norstella-light-teal);
        border-left: 4px solid var(--norstella-teal);
        border-radius: 8px;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #E6FFFA;
        border-left: 4px solid var(--norstella-teal);
        border-radius: 8px;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--norstella-teal);
        box-shadow: 0 2px 10px rgba(0, 163, 163, 0.1);
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--norstella-light-gray) 0%, white 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--norstella-teal);
        border-radius: 10px;
        background-color: var(--norstella-light-teal);
        padding: 2rem;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: var(--norstella-teal);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--norstella-light-gray);
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Set maximum upload size to 200MB (default is 200MB, but we'll be explicit)
# This should easily handle 19-page PDFs which are typically 5-20MB
MAX_UPLOAD_SIZE_MB = 200

# Configure logging
# logging.basicConfig(level=logging.INFO) # This line is now redundant as it's handled above
# logger = logging.getLogger(__name__) # This line is now redundant as it's handled above

# Disable specific loggers

logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress verbose logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

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

# Configure event loop policy for macOS
import asyncio
if sys.platform == "darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        logger.warning(f"Could not set event loop policy: {e}")

# Additional fix for torch.classes watcher issue
try:
    import torch
    # Monkey patch the problematic torch._classes.__path__ attribute
    if hasattr(torch, '_classes') and hasattr(torch._classes, '__path__'):  # type: ignore
        # Replace with a simple object that won't cause watcher issues
        class MockPath:
            def __iter__(self):
                return iter([])
            @property 
            def _path(self):
                return []
        torch._classes.__path__ = MockPath()  # type: ignore
except ImportError:
    pass  # torch not available
except Exception as e:
    logger.debug(f"Could not patch torch._classes: {e}")

from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.store import VectorStore
from src.llm.query_engine import QueryEngine
from src.report.docx_generator import ReportGenerator
from src.config import VECTORSTORE_PATH
from src.web.enhanced_analysis import enhanced_analysis_section

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
    """Set up the main page layout and title."""
    # Clean Norstella header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">🔬 Norstella AI Flash Reports</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">Strategic intelligence from your business documents</p>
    </div>
    """, unsafe_allow_html=True)

def check_api_key():
    """Check if OpenAI API key is configured."""
    if not st.session_state.openai_key:
        st.error("🔑 OpenAI API key not found!")
        st.markdown("""
        Please set your OpenAI API key:
        1. Create a `.env` file in the project root
        2. Add: `OPENAI_API_KEY=your_key_here`
        3. Restart the application
        """)
        return False
    return True

def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def upload_section():
    """Handle document uploads."""
    st.markdown("### 📁 Upload Your Documents")
    
    # Simplified controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Status:** 🟢 Ready")
    with col2:
        if st.button("🔄 Reset", help="Reset if needed"):
            for key in list(st.session_state.keys()):
                if 'uploader' in str(key):
                    del st.session_state[key]
            st.rerun()
    
    # Clean file uploader
    uploader_key = f"doc_uploader_{datetime.datetime.now().strftime('%Y%m%d_%H')}"
    
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        key=uploader_key,
        help="PDF, Word, or CSV files (max 200MB each)"
    )
    
    # Debug information for troubleshooting
    debug_mode = st.checkbox("Show debug info", help="Check this if you're having upload issues")
    
    if debug_mode:
        st.write("**Debug Information:**")
        st.write(f"- Working Directory: {os.getcwd()}")
        st.write(f"- Vectorstore Path: {VECTORSTORE_PATH}")
        st.write(f"- Max Upload Size: {MAX_UPLOAD_SIZE_MB}MB")
        st.write(f"- Uploader Key: {uploader_key}")
        st.write(f"- Session State Keys: {list(st.session_state.keys())}")
        if uploaded_files:
            st.write(f"- Files detected: {len(uploaded_files)}")
            for i, file in enumerate(uploaded_files):
                try:
                    st.write(f"  - File {i+1}: {file.name} ({file.type}) - {format_file_size(len(file.getvalue()))}")
                except Exception as e:
                    st.write(f"  - File {i+1}: {file.name} - Error reading: {str(e)}")
    
    if uploaded_files:
        # Display file information cleanly with retry logic
        st.write("**Files selected:**")
        total_size = 0
        valid_files = []
        
        for file in uploaded_files:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Get file size safely with retry
                    file.seek(0)  # Reset file pointer
                    file_content = file.getvalue()
                    file_size = len(file_content)
                    
                    if file_size == 0:
                        raise ValueError("File appears to be empty")
                    
                    total_size += file_size
                    size_str = format_file_size(file_size)
                    
                    # Check file size
                    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                        st.error(f"❌ {file.name} - {size_str} (exceeds {MAX_UPLOAD_SIZE_MB}MB limit)")
                    else:
                        st.success(f"✅ {file.name} - {size_str}")
                        valid_files.append((file, file_content))  # Store content to avoid re-reading
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        st.error(f"❌ {file.name} - Error reading file after {max_retries} attempts: {str(e)}")
                        logger.error(f"File reading error for {file.name}: {str(e)}")
                    else:
                        st.warning(f"⚠️ {file.name} - Retry {retry_count}/{max_retries}")
                        import time
                        time.sleep(0.5)  # Brief pause before retry
        
        if valid_files:
            st.write(f"**Total size:** {format_file_size(total_size)}")
            
            # Show file processing tips
            if total_size > 50 * 1024 * 1024:  # 50MB
                st.info("💡 Large files detected. Processing may take a few minutes.")
            
            # Process button with better error handling
            if st.button("Process Documents", type="primary", key="process_btn"):
                try:
                    # Create necessary directories
                    Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    # Process each file with progress tracking
                    with st.status("Processing documents...") as status:
                        processed_files = []
                        
                        for i, (file, file_content) in enumerate(valid_files):
                            try:
                                status.write(f"Processing {file.name} ({i+1}/{len(valid_files)})...")
                                
                                # Save file with better error handling
                                temp_path = temp_dir / file.name
                                
                                # Write file content (we already have it)
                                with open(temp_path, "wb") as f:
                                    f.write(file_content)
                                
                                # Verify file was written correctly
                                if not temp_path.exists():
                                    raise FileNotFoundError(f"Failed to create {temp_path}")
                                
                                actual_size = temp_path.stat().st_size
                                if actual_size != len(file_content):
                                    raise ValueError(f"File size mismatch: expected {len(file_content)}, got {actual_size}")
                                
                                processed_files.append(temp_path)
                                status.write(f"✓ Saved {file.name} ({format_file_size(actual_size)})")
                                
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")
                                logger.error(f"File processing error for {file.name}: {str(e)}")
                                continue
                        
                        if not processed_files:
                            st.error("❌ No files were successfully processed.")
                            return
                        
                        try:
                            # Load documents with timeout
                            status.write("Loading documents...")
                            loader = DocumentLoader()
                            
                            # Process files in smaller batches for better reliability
                            batch_size = 5
                            all_documents = []
                            
                            for i in range(0, len(processed_files), batch_size):
                                batch = processed_files[i:i+batch_size]
                                status.write(f"Processing batch {i//batch_size + 1}/{(len(processed_files) + batch_size - 1)//batch_size}...")
                                
                                try:
                                    batch_docs = loader.load_batch(batch)
                                    all_documents.extend(batch_docs)
                                    status.write(f"✓ Loaded {len(batch_docs)} chunks from batch")
                                except Exception as e:
                                    st.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                                    logger.error(f"Batch processing error: {str(e)}")
                                    continue
                            
                            if not all_documents:
                                st.error("❌ No documents were successfully loaded.")
                                return
                            
                            status.write(f"✓ Total loaded: {len(all_documents)} document chunks")
                            
                            # Create vector store with retry logic
                            status.write("Creating knowledge base...")
                            vectorstore = VectorStore()
                            collection_name = f"docs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    vectorstore.create_collection(all_documents, collection_name)
                                    break
                                except Exception as e:
                                    if attempt == max_retries - 1:
                                        raise e
                                    status.write(f"Retry {attempt + 1}/{max_retries} for vector store creation...")
                                    import time
                                    time.sleep(2)
                            
                            # Update session state
                            st.session_state.documents_loaded = True
                            st.session_state.vectorstore = vectorstore
                            st.session_state.collection_name = collection_name
                            
                            # Success message
                            st.success(f"""
                            ✅ Successfully processed {len(processed_files)} files
                            - Created {len(all_documents)} text chunks
                            - Collection: {collection_name}
                            """)
                            
                            # Guide to next step
                            st.info("👉 Go to the AI Chat Analysis tab to start conversing with your documents")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing documents: {str(e)}")
                            logger.error(f"Document processing error: {str(e)}")
                            
                            # Provide specific troubleshooting advice
                            st.markdown("""
                            **Troubleshooting Steps:**
                            1. Try uploading fewer files at once
                            2. Check that files aren't corrupted
                            3. Try refreshing the page and uploading again
                            4. Use the Reset Upload button above
                            """)
                    
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    logger.error(f"Unexpected error in upload section: {str(e)}")
                
                finally:
                    # Cleanup with better error handling
                    try:
                        if 'processed_files' in locals():
                            for path in processed_files:
                                if path.exists():
                                    path.unlink()
                        if 'temp_dir' in locals() and temp_dir.exists():
                            temp_dir.rmdir()
                    except Exception as e:
                        logger.warning(f"Cleanup error (non-critical): {str(e)}")
        else:
            st.warning("⚠️ No valid files selected. Please check file sizes and formats.")
    
    else:
        # Simple help text
        st.info("📁 Select PDF, DOCX, or CSV files to get started")
        
        # Simplified sidebar help
        st.sidebar.markdown("""
        ### 🔬 Norstella AI Assistant
        
        **Supported Files:**
        PDF, Word, CSV (up to 200MB)
        
        **AI Features:**
        • Smart document analysis
        • Multiple conversation styles  
        • Strategic report generation
        • Conversational memory
        """)
        
        # Enhanced troubleshooting tips
        with st.expander("🔧 Troubleshooting Upload Issues"):
            st.markdown("""
            **Common Solutions:**
            
            **Files not appearing:**
            1. Click "Reset Upload" button above
            2. Refresh the page (F5 or Ctrl+R)
            3. Try "Clear Cache" button
            4. Check file format (PDF, DOCX, CSV only)
            
            **Upload fails or times out:**
            1. Check file size (must be under 200MB)
            2. Try uploading one file at a time
            3. Ensure stable internet connection
            4. Close other browser tabs using bandwidth
            
            **Files disappear after upload:**
            1. This is a known Streamlit issue with large files
            2. Try smaller files first
            3. Use "Reset Upload" and try again
            4. Check the debug info for error details
            
            **Still having issues?**
            - Enable "Show debug info" checkbox
            - Check browser console for errors (F12)
            - Try a different browser
            """)


def analysis_section():
    """Handle document analysis using enhanced conversational interface."""
    # Use the enhanced conversational analysis instead of basic query interface
    enhanced_analysis_section()


def report_section():
    """Handle report generation."""
    if not st.session_state.documents_loaded:
        st.info("⚠️ Upload documents first")
        return
    
    st.markdown("### 📊 Generate Flash Report")
    
    # Simple report configuration
    st.markdown("**Configuration**")
    
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
    
    # Simplified section selection
    st.markdown("**Report Sections**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_exec_summary = st.checkbox("🧾 Executive Summary", value=True)
        include_company = st.checkbox("🧭 Company Overview", value=True)
        include_offerings = st.checkbox("📦 Core Offerings", value=True)
    
    with col2:
        include_market = st.checkbox("📈 Market Position", value=True)
        include_insights = st.checkbox("🧠 Strategic Insights", value=True)
    
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
                    subtitle=f"Generated on {report_date}"
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
                    if content and len(content.strip()) > 50:
                        generator.add_section("Executive Summary", content)
                        status.write("✓ Executive summary generated")
                    else:
                        status.write("⚠️ Executive summary skipped (insufficient content)")
                
                if include_company:
                    status.write("Analyzing company overview...")
                    context = vs.query_collection(
                        "Extract company information, business model, and recent strategic moves"
                    )
                    content = engine.generate_section_content("company_overview", context)
                    if content and len(content.strip()) > 50:
                        generator.add_section("Company Overview", content)
                        status.write("✓ Company overview generated")
                    else:
                        status.write("⚠️ Company overview skipped (insufficient content)")
                
                if include_offerings:
                    status.write("Analyzing core offerings...")
                    context = vs.query_collection(
                        "Extract information about products, services, platforms, and recent launches or changes"
                    )
                    content = engine.generate_section_content("core_offerings", context)
                    if content and len(content.strip()) > 50:
                        generator.add_section("Core Offerings", content)
                        status.write("✓ Core offerings generated")
                    else:
                        status.write("⚠️ Core offerings skipped (insufficient content)")
                
                if include_market:
                    status.write("Analyzing market position...")
                    context = vs.query_collection(
                        "Analyze market position, competitors, differentiators, and market trends"
                    )
                    content = engine.generate_section_content("market_position", context)
                    if content and len(content.strip()) > 50:
                        generator.add_section("Market Position", content)
                        status.write("✓ Market position generated")
                    else:
                        status.write("⚠️ Market position skipped (insufficient content)")
                
                if include_insights:
                    status.write("Generating strategic insights...")
                    context = vs.query_collection(
                        "Generate strategic insights, implications, and recommendations"
                    )
                    content = engine.generate_section_content("strategic_insights", context)
                    if content and len(content.strip()) > 50:
                        generator.add_section("Key Strategic Insights", content)
                        status.write("✓ Strategic insights generated")
                    else:
                        status.write("⚠️ Strategic insights skipped (insufficient content)")
                
                # Generate report in memory
                status.write("Finalizing report...")
                report_filename = f"flash_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                
                status.update(label="✅ Report generated successfully!", state="complete")
            
            # Provide download using BytesIO (Streamlit Cloud compatible)
            st.download_button(
                label="📄 Download Report",
                data=generator.get_docx_bytes(),
                file_name=report_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            st.error("Error generating report. Please try again.")


def main():
    """Main application function."""
    init_session_state()
    setup_page()
    
    if not check_api_key():
        return
    
    # Clean navigation
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🔬 AI Analysis", "📊 Report"])
    
    with tab1:
        upload_section()
    
    with tab2:
        analysis_section()
    
    with tab3:
        report_section()


if __name__ == "__main__":
    main() 
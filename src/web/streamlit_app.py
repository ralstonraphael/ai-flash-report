"""
Streamlit web interface for the Flash Report Generator.
"""
import streamlit as st
from pathlib import Path
import tempfile
import os
import logging
import warnings

# Filter out warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")

from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.store import VectorStore
from src.llm.query_engine import QueryEngine, QueryIntent
from src.report.docx_generator import ReportGenerator
from src.config import VECTORSTORE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


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


def setup_page():
    """Configure page settings and layout."""
    st.set_page_config(
        page_title="Flash Report Generator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Flash Report Generator")


def file_uploader_section():
    """Handle file uploads and processing."""
    # Check for OpenAI API key
    if not st.session_state.openai_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.openai_key = api_key
        else:
            st.info("Please enter your OpenAI API key to continue")
            return
    
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, CSV)",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Selected {len(uploaded_files)} files")
        with col2:
            process_button = st.button("Process Documents")
        
        if process_button:
            with st.status("Processing documents...") as status:
                # Save uploaded files temporarily
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                try:
                    # Save files
                    for uploaded_file in uploaded_files:
                        temp_path = Path(temp_dir) / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        file_paths.append(temp_path)
                        logger.info(f"Saved: {temp_path}")
                    
                    # Process documents
                    status.write("Loading documents...")
                    loader = DocumentLoader()
                    documents = loader.load_batch(file_paths)
                    
                    status.write("Creating embeddings...")
                    vectorstore = VectorStore()
                    collection_name = "uploaded_docs_" + str(hash("".join([str(f.name) for f in uploaded_files])))
                    vectorstore.create_collection(documents, collection_name)
                    
                    # Update session state
                    st.session_state.documents_loaded = True
                    st.session_state.vectorstore = vectorstore
                    st.session_state.collection_name = collection_name
                    
                    status.update(label="✅ Documents processed!", state="complete")
                    
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    st.error("Error processing documents. Please check file formats and try again.")
                    
                finally:
                    # Cleanup
                    for path in file_paths:
                        try:
                            os.remove(path)
                        except Exception as e:
                            logger.error(f"Error removing {path}: {e}")
                    try:
                        os.rmdir(temp_dir)
                    except Exception as e:
                        logger.error(f"Error removing temp dir: {e}")


def query_section():
    """Handle query input and response generation."""
    if not st.session_state.documents_loaded:
        return
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_area("Enter your question")
    with col2:
        query_type = st.selectbox(
            "Query Type",
            ["Summary", "Specific Question", "Data Extraction"]
        )
        generate_button = st.button("Generate Response")
    
    if query and generate_button:
        with st.status("Generating response...") as status:
            try:
                # Map query type to intent
                intent_map = {
                    "Summary": QueryIntent.SUMMARY,
                    "Specific Question": QueryIntent.SPECIFIC_QUESTION,
                    "Data Extraction": QueryIntent.DATA_EXTRACTION
                }
                
                # Get context and generate response
                vs = st.session_state.vectorstore
                status.write("Finding relevant information...")
                context = vs.query_collection(query)
                
                status.write("Generating response...")
                engine = QueryEngine()
                response = engine.generate_response(
                    query=query,
                    context=context,
                    intent=intent_map[query_type]
                )
                
                status.update(label="✅ Response ready!", state="complete")
                
                # Show response
                st.write(response)
                
                # Show metrics in a container instead of an expander
                st.container()
                st.subheader("Quality Metrics")
                metrics = engine.evaluate_response(query, response, context)
                st.json(metrics)
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                st.error("Error generating response. Please try again.")


def export_section():
    """Handle report export functionality."""
    if not st.session_state.documents_loaded:
        return
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Report Title", "Flash Intelligence Report")
    with col2:
        subtitle = st.text_input("Subtitle (optional)")
    
    if st.button("Generate Report"):
        with st.status("Generating report...") as status:
            try:
                # Initialize report generator
                status.write("Creating report...")
                generator = ReportGenerator()
                
                # Add title page
                generator.add_title_page(
                    title=title,
                    subtitle=subtitle,
                    logo_path=str(Path("templates/Images/Norstella_color_positive_RGB_(2).png"))
                )
                
                # Generate content
                vs = st.session_state.vectorstore
                engine = QueryEngine()
                
                # Executive summary
                status.write("Generating executive summary...")
                summary_context = vs.query_collection("Generate a comprehensive executive summary")
                summary = engine.generate_response(
                    query="Generate a detailed executive summary that covers key points, trends, and implications",
                    context=summary_context,
                    intent=QueryIntent.SUMMARY
                )
                generator.add_section("Executive Summary", summary)
                
                # Key findings
                status.write("Extracting key findings...")
                findings_context = vs.query_collection("Extract key findings, metrics, and insights")
                findings = engine.generate_response(
                    query="Extract and organize key findings, important metrics, and strategic insights",
                    context=findings_context,
                    intent=QueryIntent.DATA_EXTRACTION
                )
                generator.add_section("Key Findings", findings)
                
                # Save and offer download
                status.write("Preparing download...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    generator.save(tmp.name)
                    
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "📥 Download Report",
                            data=f.read(),
                            file_name="flash_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    os.unlink(tmp.name)
                    
                status.update(label="✅ Report ready!", state="complete")
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                st.error("Error generating report. Please try again.")


def main():
    """Main application entry point."""
    init_session_state()
    setup_page()
    
    # Create layout
    file_uploader_section()
    query_section()
    export_section()


if __name__ == "__main__":
    main() 
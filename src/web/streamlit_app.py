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
import pandas as pd

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
from src.generation.visualizer import DataVisualizer

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
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None


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
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your source documents",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, CSV"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        file_names = [f.name for f in uploaded_files]
        
        # Show file list
        with st.expander("📋 Uploaded Files", expanded=True):
            for i, name in enumerate(file_names, 1):
                st.text(f"{i}. {name}")
        
        # Process button
        if st.button("Process Documents", type="primary"):
            with st.status("Processing documents...") as status:
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                try:
                    # Save files
                    for file in uploaded_files:
                        temp_path = Path(temp_dir) / file.name
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())
                        file_paths.append(temp_path)
                        status.write(f"Saved: {file.name}")
                    
                    # Process documents
                    status.write("Analyzing documents...")
                    loader = DocumentLoader()
                    documents = loader.load_batch(file_paths)
                    
                    status.write("Creating knowledge base...")
                    vectorstore = VectorStore()
                    collection_name = "uploaded_docs_" + str(hash("".join([str(f.name) for f in uploaded_files])))
                    vectorstore.create_collection(documents, collection_name)
                    
                    # Update session state
                    st.session_state.documents_loaded = True
                    st.session_state.vectorstore = vectorstore
                    st.session_state.collection_name = collection_name
                    
                    status.update(label="✅ Documents processed successfully!", state="complete")
                    
                    # Show success message with stats
                    st.success(f"""
                    Processed {len(uploaded_files)} documents:
                    - Created {len(documents)} text chunks
                    - Documents are ready for analysis
                    """)
                    
                    # Guide user to next step
                    st.info("👉 Go to the Analysis tab to start exploring your documents")
                    
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
                
                # Store the response for visualization
                st.session_state.analysis_result = response
                
                # Extract numerical data if available
                if query_type == "Data Extraction":
                    visualizer = DataVisualizer()
                    extracted_data = visualizer.extract_numerical_data(response)
                    if extracted_data:
                        st.session_state.extracted_data = extracted_data
                
                # Clear progress message
                progress_placeholder.empty()
                
                # Show results
                st.markdown("### Analysis Results")
                st.write(response)
                
                # Show visualization option if numerical data is available
                if st.session_state.extracted_data:
                    st.info("📊 Numerical data detected! Go to the Visualization tab to create charts.")
                
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


def visualization_section():
    """Handle data visualization interface."""
    if not st.session_state.extracted_data:
        st.info("⚠️ No numerical data available. Run a data extraction query first!")
        return
    
    st.subheader("📊 Data Visualization")
    
    # Show raw data
    with st.expander("View Raw Data"):
        df = pd.DataFrame(st.session_state.extracted_data)
        if not df.empty:
            st.dataframe(df)
    
    # Get visualization suggestion
    visualizer = DataVisualizer()
    suggestion = visualizer.suggest_visualization(st.session_state.extracted_data)
    
    if suggestion.get('data_group'):
        # Visualization controls
        st.markdown("### Chart Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["bar", "line", "pie", "scatter"],
                index=["bar", "line", "pie", "scatter"].index(suggestion["chart_type"]),
                help=suggestion["explanation"]
            )
            
            title = st.text_input("Chart Title", value=suggestion["title"])
        
        with col2:
            x_axis = st.text_input("X-Axis Label", value=suggestion["x_axis"])
            y_axis = st.text_input("Y-Axis Label", value=suggestion["y_axis"])
        
        # Create and display chart
        try:
            fig = visualizer.create_chart(
                suggestion['data_group'],
                chart_type,
                title,
                x_axis,
                y_axis
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.markdown("### Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as HTML
                html_str = fig.to_html()
                st.download_button(
                    label="Download as HTML",
                    data=html_str,
                    file_name="visualization.html",
                    mime="text/html"
                )
            
            with col2:
                # Download as PNG
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name="visualization.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Try selecting different metrics or a different chart type.")
    else:
        st.warning("No compatible data groups found for visualization. Try extracting different metrics.")


def report_section():
    """Handle report generation."""
    if not st.session_state.documents_loaded:
        st.info("⚠️ Please upload and process documents first")
        return
    
    st.subheader("📊 Flash Report Generation")
    
    # Report configuration
    st.markdown("### Report Configuration")
    st.info("📄 Reports are automatically formatted to fit on one page")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Report Title",
            value="Flash Report",
            help="Enter the title for your report"
        )
        
        include_charts = st.checkbox(
            "Include Visualizations",
            value=False,
            help="Include any visualizations created in the Visualization tab"
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
    st.caption("Select sections to include (content will be automatically sized to fit one page)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_exec_summary = st.checkbox("🧾 Executive Summary", value=True,
            help="Brief overview of key points and changes (2-3 sentences)")
        include_company = st.checkbox("🧭 Company Overview", value=True,
            help="Essential company info (2-3 bullet points)")
        include_offerings = st.checkbox("📦 Core Offerings", value=True,
            help="Key products/services (3-4 bullet points)")
    
    with col2:
        include_market = st.checkbox("📈 Market Position", value=True,
            help="Competitive landscape (2-3 key points)")
        include_insights = st.checkbox("🧠 Key Strategic Insights", value=True,
            help="Strategic recommendations (3-4 bullet points)")
        include_metrics = st.checkbox("📊 Key Metrics", value=True,
            help="Important numbers (5-7 metrics)")
    
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
                    generator.add_section("Executive Summary", content, max_length=200)
                
                if include_company:
                    status.write("Analyzing company overview...")
                    context = vs.query_collection(
                        "Extract company information, business model, and recent strategic moves"
                    )
                    content = engine.generate_section_content("company_overview", context)
                    generator.add_section("Company Overview", content, max_length=150)
                
                if include_offerings:
                    status.write("Analyzing core offerings...")
                    context = vs.query_collection(
                        "Extract information about products, services, platforms, and recent launches or changes"
                    )
                    content = engine.generate_section_content("core_offerings", context)
                    generator.add_section("Core Offerings", content, max_length=150)
                
                if include_market:
                    status.write("Analyzing market position...")
                    context = vs.query_collection(
                        "Analyze market position, competitors, differentiators, and market trends"
                    )
                    content = engine.generate_section_content("market_position", context)
                    generator.add_section("Market Position", content, max_length=150)
                
                if include_metrics:
                    status.write("Extracting key metrics...")
                    if st.session_state.extracted_data:
                        content = "Key Metrics:\n\n"
                        # Limit to top 5-7 metrics
                        for item in st.session_state.extracted_data[:7]:
                            content += f"• {item['label']}: {item['value']}{item['unit']}"
                            if item.get('time'):
                                content += f" ({item['time']})"
                            content += "\n"
                        generator.add_section("Key Metrics", content, max_length=200)
                        
                        if include_charts and 'visualization_fig' in st.session_state:
                            generator.add_chart_from_file(
                                st.session_state.visualization_fig,
                                title="Key Metrics Visualization",
                                caption="Visual representation of key metrics and trends"
                            )
                
                if include_insights:
                    status.write("Generating strategic insights...")
                    context = vs.query_collection(
                        "Generate strategic insights, implications, and recommendations"
                    )
                    content = engine.generate_section_content("strategic_insights", context)
                    generator.add_section("Strategic Insights", content, max_length=200)
                
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
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Analysis", "Visualization", "Report"])
    
    with tab1:
        upload_section()
    
    with tab2:
        analysis_section()
    
    with tab3:
        visualization_section()
    
    with tab4:
        report_section()


if __name__ == "__main__":
    main() 
# 📊 Flash Report Generator

An AI-powered competitive intelligence tool that generates insightful reports from corporate documents using RAG (Retrieval Augmented Generation).

## 🌟 Features

- **Multi-format Document Processing**: Support for PDF, DOCX, CSV, and more
- **Intelligent Query Understanding**: Automatic intent detection for better responses
- **Advanced Retrieval**: Uses MMR (Maximal Marginal Relevance) for diverse, relevant results
- **Quality Metrics**: Built-in evaluation of response accuracy and relevance
- **Professional Reports**: Customizable DOCX export with styling and branding
- **User-friendly Interface**: Streamlit web UI for easy interaction

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/flash-report-generator.git
cd flash-report-generator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application

1. Start the Streamlit interface:

```bash
streamlit run src/web/streamlit_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
flash-report-generator/
├── src/
│   ├── ingestion/           # Document loading and processing
│   ├── vectorstore/         # Vector database management
│   ├── llm/                 # LLM query handling
│   ├── report/             # Report generation
│   ├── web/                # Streamlit interface
│   └── config.py           # Configuration settings
├── templates/              # Report templates and assets
├── notebooks/             # Development notebooks
├── tests/                 # Test suite
└── requirements.txt       # Project dependencies
```

## 🔧 Configuration

Key settings in `config.py`:

- Document chunk size and overlap
- OpenAI model selection
- Vector store settings
- Report styling options
- Evaluation metrics

## 💡 Usage Examples

1. **Upload Documents**:

   - Select multiple files (PDF, DOCX, CSV)
   - Click "Process Documents" to analyze

2. **Ask Questions**:

   - Choose query type (Summary, Specific Question, Data Extraction)
   - Enter your query
   - View response and quality metrics

3. **Generate Reports**:
   - Set report title and options
   - Click "Generate DOCX Report"
   - Download the formatted document

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Vector storage by [ChromaDB](https://github.com/chroma-core/chroma)
- Document processing with [Docling](https://github.com/docling/docling)

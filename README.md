# 📊 AI Flash Report Generator

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
git clone https://github.com/yourusername/ai-flash-report.git
cd ai-flash-report
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

#### Local Development

```bash
streamlit run src/web/streamlit_app.py
```

#### Using the run script

```bash
python run_app.py
```

Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment to Streamlit Cloud

### Prerequisites for Deployment

1. **GitHub Repository**: Push your code to a GitHub repository
2. **OpenAI API Key**: Have your OpenAI API key ready

### Deployment Steps

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Connect your GitHub account**
3. **Deploy new app**:
   - Repository: `your-username/ai-flash-report`
   - Branch: `main`
   - Main file path: `src/web/streamlit_app.py`
4. **Configure secrets**:
   - Go to your app settings
   - Add secret: `OPENAI_API_KEY = "your_actual_api_key_here"`
5. **Deploy**

### Environment Configuration

The app supports both local `.env` files and Streamlit Cloud secrets:

**Local Development:**

```bash
# .env file
OPENAI_API_KEY=your_key_here
```

**Streamlit Cloud:**

```toml
# In app settings -> Secrets
OPENAI_API_KEY = "your_key_here"
```

## 📁 Project Structure

```
ai-flash-report/
├── src/
│   ├── ingestion/           # Document loading and processing
│   ├── vectorstore/         # Vector database management
│   ├── llm/                 # LLM query handling
│   ├── report/             # Report generation
│   ├── web/                # Streamlit interface
│   └── config.py           # Configuration settings
├── templates/              # Report templates and assets
├── .streamlit/            # Streamlit configuration
├── requirements.txt       # Project dependencies
├── run_app.py            # Application runner
└── README.md             # This file
```

## 🔧 Configuration

Key settings in `src/config.py`:

- Document chunk size and overlap
- OpenAI model selection
- Vector store settings
- Report styling options
- Evaluation metrics

## 💡 Usage

1. **Upload Documents**:

   - Select multiple files (PDF, DOCX, CSV)
   - Click "Process Documents" to analyze

2. **Ask Questions**:

   - Choose query type (Summary, Specific Question, Data Extraction)
   - Enter your query
   - View response and quality metrics

3. **Generate Reports**:
   - Set report title and options
   - Click "Generate Report"
   - Download the formatted DOCX document

## 🚨 Important Notes

- **File Size Limit**: 200MB per file
- **Supported Formats**: PDF, DOCX, CSV
- **OpenAI API**: Requires valid API key with sufficient credits
- **Memory Usage**: Large documents may require significant processing time

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenAI API key is valid and has credits
2. **File Upload Issues**: Try smaller files or refresh the page
3. **Processing Timeouts**: Large documents may take several minutes
4. **Memory Issues**: Restart the app if it becomes unresponsive

### For Streamlit Cloud Deployment

1. **Build Errors**: Check that all dependencies are in requirements.txt
2. **Secret Issues**: Verify API key is correctly set in app settings
3. **Performance**: Streamlit Cloud has resource limits for free tier

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Vector storage by [Pinecone](https://www.pinecone.io/)
- Document processing with [Docling](https://github.com/docling/docling)
- UI powered by [Streamlit](https://streamlit.io/)

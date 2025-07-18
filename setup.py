from setuptools import setup, find_packages

setup(
    name="flash-report",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "src": ["templates/*", "templates/Images/*"],
    },
    install_requires=[
        "langchain>=0.3.25",
        "langchain-community>=0.3.25",
        "langchain-core>=0.3.65",
        "langchain-openai>=0.3.22",
        "openai>=1.86.0",
        "python-dotenv>=1.1.0",
        "python-docx>=1.1.2",
        "pypdf>=5.6.0",
        "pandas>=2.3.0",
        "docling>=2.36.1",
        "chromadb==0.4.22",
        "streamlit>=1.45.1",
    ],
    python_requires=">=3.8",
) 
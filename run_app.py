"""
Run the Flash Report Generator Streamlit app.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.bootstrap
    
    # Get the path to the Streamlit app
    app_path = project_root / "src" / "web" / "streamlit_app.py"
    
    # Run the app
    streamlit.web.bootstrap.run(str(app_path), "", [], []) 
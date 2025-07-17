#!/bin/bash

# Start ChromaDB server in the background
chroma run --host 0.0.0.0 --port 8000 --path /tmp/chromadb &

# Wait for ChromaDB server to start
sleep 5

# Start Streamlit app
streamlit run src/web/streamlit_app.py 
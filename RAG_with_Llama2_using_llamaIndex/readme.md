# RAG with LLaMA 2 using LlamaIndex

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Meta's LLaMA 2 language model and the LlamaIndex framework. It enables intelligent querying over custom PDF documents by combining language generation with vector-based document retrieval.

## Features

* Uses LLaMA 2 for language generation
* Indexes documents using LlamaIndex
* Ingests PDF files from a local directory
* Interactive querying via a Streamlit UI
* Modular and easy-to-extend codebase

## Project Structure

* `app.py`: Streamlit-based user interface
* `ingest.py`: Loads and indexes documents
* `query_engine.py`: Configures the query engine using LlamaIndex
* `utils.py`: Utility functions for PDF reading
* `requirements.txt`: Python dependencies
* `data/`: Folder containing source PDF documents

## Installation

1. Clone the repository:

   ````bash
   git clone https://github.com/ashdrift231/GenAI_projects.git
   cd GenAI_projects/RAG_with_Llama2_using_llamaIndex
   ````


# RAG with LangChain, AstraDB, and NVIDIA AI Endpoints

## Project Overview

The goal of this project is to create a RAG system that can answer questions based on the content of a specific PDF document/documents. It leverages vector embeddings, a vector database, and a large language model to achieve this.

## Features

*   **PDF Processing:** Loads and processes a PDF document using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
*   **Vector Embeddings:** Generates vector embeddings for document chunks using `NVIDIAEmbeddings`.
*   **Vector Database:** Stores and retrieves document embeddings using `AstraDBVectorStore`.
*   **Retrieval Augmented Generation (RAG):** Implements a RAG pipeline with LangChain and LangGraph to retrieve relevant context and generate answers using a `ChatNVIDIA` model (meta/llama3-70b-instruct).
*   **API Key Management:** Demonstrates how to securely manage API keys using Google Colab's `userdata`.

## Getting Started

### Prerequisites

*   API keys for NVIDIA AI Endpoints and AstraDB. You will need to set up an AstraDB database and obtain the API endpoint and application token.

### Installation

1.  Open the provided Google Colab notebook.
2.  Run the installation cells:

### Setup

1.  In your Google Colab notebook, go to the "Secrets" tab (the key icon on the left sidebar).
   ![{BB0F73EF-27F8-42B3-B0B1-9FBA59BC5532}](https://github.com/user-attachments/assets/ed42dbbc-1a8f-4b14-9224-4183f19fcdd7)
2.  Add the following secrets:
    *   `llm_projects`: Your NVIDIA AI Endpoints API key.
    *   `ASTRA_DB_API_ENDPOINT`: Your AstraDB API Endpoint.
    *   `ASTRA_DB_APPLICATION_TOKEN`: Your AstraDB Application Token.
    *   `LANGSMITH_API_KEY`: Your LangSmith API Key (if using LangSmith tracing).
3.  Ensure the environment variables in the notebook are set correctly to read from `userdata`.
4.  Upload your PDF document (e.g., `22365_19_Agents_v8.pdf`) to the Colab environment.

### Running the Notebook

1.  Run all the cells in the Google Colab notebook in order.
2.  Modify the question in the final cell to ask something about the content of your uploaded PDF

## Code Structure

The notebook follows a typical flow for a RAG application:

1.  **Dependency Installation:** Installing required Python packages.
2.  **API Key Setup:** Loading API keys from Colab `userdata` and setting environment variables.
3.  **Model and Embeddings Initialization:** Setting up the ChatNVIDIA model and NVIDIA Embeddings.
4.  **Data Loading and Processing:** Loading the PDF, splitting it into chunks, and creating embeddings.
5.  **Vector Store Setup:** Initializing the AstraDB vector store and adding the document chunks.
6.  **RAG Graph Definition:** Defining the retrieval and generation steps using LangGraph.
7.  **Inference:** Running the RAG pipeline with a user query.

## Technologies Used

*   [**LangChain**](https://www.langchain.com/): A framework for developing applications powered by language models.
*   [**AstraDB**](https://www.datastax.com/products/datastax-astra-db): A serverless cloud database with vector search capabilities.
*   [**NVIDIA AI Endpoints**](https://build.nvidia.com/): Provides access to large language models and embedding models.
*   [**LangGraph**](https://langchain-ai.github.io/langgraph/): A library for building stateful, multi-actor applications with language models.
*   **PyPDF2:** A library for working with PDF files.

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository and submit a pull request.

## Acknowledgements

*   The LangChain documentation and examples.
*   The AstraDB documentation.
*   NVIDIA AI Endpoints documentation.


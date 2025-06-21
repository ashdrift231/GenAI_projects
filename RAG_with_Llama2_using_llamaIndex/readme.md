# 🧠 Retrieval-Augmented Generation (RAG) with Llama2 using LlamaIndex

This project demonstrates a simple end-to-end pipeline for building a Retrieval-Augmented Generation (RAG) system using **LlamaIndex** and **Meta's Llama 2** model via Hugging Face. It allows you to load documents, index them, and generate responses to natural language queries using contextual information from your documents.

---

## 🚀 Features

- Load and chunk text documents from a local folder
- Create a vector-based index using LlamaIndex
- Use Hugging Face's `Llama-2-7b-chat-hf` model for response generation
- Query your documents in a question-answering fashion
- Jupyter Notebook demonstration included

---

## 📁 Project Structure

RAG_with_Llama2_using_llamaIndex/
├── data/ # Folder containing source documents
├── RAG_with_Llama2.ipynb # Main notebook for running the pipeline
├── rag.py # Optional script for CLI usage (if implemented)
├── requirements.txt # Python dependencies
└── README.md # Project documentation (you’re here!)

yaml
Copy
Edit

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/ashdrift231/GenAI_projects
cd GenAI_projects/RAG_with_Llama2_using_llamaIndex
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies

🔒 Note: You’ll need access to Meta’s Llama 2 weights via Hugging Face. Make sure you’re authenticated.

## 🛠️ Usage Instructions
Add .txt or .md files to the data/ folder.

Run the Jupyter notebook:

The notebook walks you through:

-Reading and chunking documents
-Creating an index
-Loading the Llama 2 model
-Asking questions and receiving context-based answers

🧪 Sample Code Overview
```bash
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor
from transformers import LlamaTokenizer, LlamaForCausalLM

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Initialize tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Set up LLM predictor
llm_predictor = LLMPredictor(llm=model, tokenizer=tokenizer)
index.set_llm_predictor(llm_predictor)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main idea of the document?")
print(response)
```

✅ To-Do & Improvements
-Add persistent index storage using storage_context
-Try other embedding models (e.g., sentence-transformers)
-Deploy as a REST API using FastAPI
-Streamlit or Gradio UI for public interaction

📚 References
-LlamaIndex Documentation
-Llama 2 Models on Hugging Face
-RAG Architecture Overview

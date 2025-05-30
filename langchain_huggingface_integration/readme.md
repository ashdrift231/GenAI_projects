# Langchain and Hugging Face Integration Example

This notebook demonstrates how to integrate Langchain with Hugging Face models, utilizing both the Hugging Face Inference API (via `HuggingFaceEndpoint`) and local model loading with quantization (via `HuggingFacePipeline`).

## Setup

1.  **Install Libraries:** The notebook begins by installing the necessary libraries:
    *   `langchain`: The core library for building LLM applications.
    *   `langchain-huggingface`: Provides integration between Langchain and Hugging Face models.
    *   `transformers`: Hugging Face's library for accessing pre-trained models.
    *   `accelerate`: Helps speed up training and inference for large models.
    *   `bitsandbytes`: Used for quantization, reducing model size and potentially speeding up inference.
    *   `huggingface_hub`: For interacting with the Hugging Face Hub.

2.  **Authenticate with Hugging Face:**
    *   Retrieve your Hugging Face API token from Colab's user data secrets.
    *   Set the API token as an environment variable for authentication.


*Make sure you have stored your Hugging Face API token in Colab's secrets with the name 'HF_llm'.*

## Hugging Face Integration with Langchain

This notebook illustrates two primary ways to interact with Hugging Face models using Langchain:

### 1. `HuggingFaceEndpoint`: Using the Hugging Face Inference API

The `HuggingFaceEndpoint` class allows you to use a model hosted on the Hugging Face Inference API directly from your notebook. This is convenient as it doesn't require downloading the model weights locally.

*   **How it works:** When you use `HuggingFaceEndpoint`, Langchain sends your prompt to the specified model hosted on Hugging Face's servers via an API call. The response is then returned to your notebook.
*   **When to use it:** This is suitable for quick testing, using large models that might not fit in your local memory, or when you prefer not to manage model weights locally.
*   **In the notebook:**
    *   An instance of `HuggingFaceEndpoint` is created, specifying the `repo_id` (model identifier), parameters like `max_new_tokens` and `temperature`, and your API token for authentication.
    *   The `llm.invoke()` method sends the prompt to the API endpoint.
 
### 2. `HuggingFacePipeline`: Loading the Model Locally

The `HuggingFacePipeline` class allows you to load a Hugging Face model directly into your notebook's environment. This gives you more control and can be necessary for certain workflows, especially when combined with techniques like quantization.

*   **How it works:** `HuggingFacePipeline.from_model_id` downloads the specified model weights and configuration to your local runtime (in this case, the Colab instance). It then sets up a Hugging Face `pipeline` for a specific task (like text generation), which runs locally.
*   **Quantization:** The example demonstrates using `BitsAndBytesConfig` with `HuggingFacePipeline`. Quantization is a technique to reduce the precision of the model weights (e.g., from 32-bit to 4-bit) to significantly lower memory requirements and potentially speed up inference on compatible hardware. When the model is loaded using `HuggingFacePipeline`, the specified quantization configuration is applied.
*   **When to use it:** Use this when you need more control over the model loading process, want to apply techniques like quantization, or when you need faster inference for repeated calls by avoiding API latency.
*   **In the notebook:**
    *   A `BitsAndBytesConfig` is defined to specify 4-bit quantization.
    *   `HuggingFacePipeline.from_model_id` loads the model locally, applying the quantization configuration (`model_kwargs`). Pipeline-specific arguments like `max_new_tokens` are also set.
    *   The loaded pipeline (`llm2`) is wrapped in `ChatHuggingFace` to handle conversational prompts.
    *   `ChatHuggingFace(llm=llm2, model_id=repo_id).invoke(messages)` runs the prompt through the locally loaded and quantized model.


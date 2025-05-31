# YouTube Video Summarizer

This notebook provides a simple Streamlit application to summarize YouTube videos using the Google Gemini Pro model.

## Features

- Extracts the transcript of a given YouTube video.
- Utilizes the Gemini Pro model to generate a concise summary of the video transcript.
- Displays the video thumbnail.

## Setup

1.  **Install Dependencies:** Run the cell containing `!pip install -r /content/requirements.txt` to install the required libraries (`streamlit`, `google-generativeai`, `youtube-transcript-api`, `python-dotenv`).
2.  **Configure API Key:**
    *   Replace `"your_api_key"` in the `genai.configure(api_key="your_api_key")` cell with your actual Google Gemini API key.
    *   Alternatively, you can use environment variables by uncommenting the `from dotenv import load_dotenv` and `load_dotenv()` lines and setting your API key in a `.env` file in the Colab environment.
3.  **Run the Streamlit App:** Execute the remaining cells to run the Streamlit application within Colab. A public URL will be provided to access the app.

## Usage

1.  Enter the URL of the YouTube video you want to summarize in the text input field.
2.  Click the "Summarize" button.
3.  The video thumbnail and the generated summary will be displayed below.

## Requirements

The `requirements.txt` file should contain the following:
youtube_transcript_api
streamlit
google-generativeai
pathlib

## Code Structure

-   **Import Libraries:** Imports necessary libraries like `streamlit`, `google.generativeai`, and `youtube_transcript_api`.
-   **API Configuration:** Sets up the Google Gemini API key.
-   **Prompt Definition:** Defines the prompt used for the Gemini model to guide the summarization.
-   **Streamlit App:** Sets up the Streamlit interface with a title, text input for the URL, and a button.
-   **Transcript Extraction:** A function `extract_transcript` to fetch the video transcript.
-   **Summary Generation:** A function `generate_summary` to use the Gemini model for summarization.
-   **Display Results:** Shows the thumbnail and the generated summary.

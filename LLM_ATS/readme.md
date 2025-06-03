# Smart ATS Resume Analyzer

This project is a Streamlit application that acts as a Smart Application Tracking System (ATS) to analyze resumes based on a given job description. It utilizes the NVIDIA AI Endpoints to compare the resume text against the job description and provide a percentage match, missing keywords, and a profile summary.

## Features

* Upload your resume in PDF format.
* Paste the job description you are targeting.
* Get a percentage match between your resume and the job description.
* Identify missing keywords in your resume.
* Receive a profile summary based on the analysis.

## Prerequisites

* Google Colab environment or a similar Jupyter Notebook environment.
* An NVIDIA API key.

## Setup and Running

1. **Open the Notebook:** Open the provided Google Colab notebook.
2. **Install Dependencies:** Run the first code cell to install the required Python libraries and localtunnel.
3. **Add Your NVIDIA API Key:** In the second code cell (the one with `%%writefile app.py`), replace `"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"` with your actual NVIDIA API key. It is recommended to use environment variables for better security in a production environment.
4. **Write the Streamlit App:** Run the second code cell to create the `app.py` file with the Streamlit application code.
5. **Run the App:** Execute the third code cell. This will:
    * Start the Streamlit application.
    * Use localtunnel to create a publicly accessible URL for your application.
    * Display the localtunnel URL and your public IP address.
6. **Access the App:** Click on the localtunnel URL provided in the output of the third cell to open the Streamlit application in your web browser.

## Usage

1. In the Streamlit application, paste the Job Description into the provided text area.
2. Click on "Upload Your Resume" and select your resume PDF file.
3. Click the "Submit" button.
4. The application will display the analysis results, including the JD match percentage, missing keywords, and profile summary.

## Code Overview

* **Dependency Installation:** The first cell installs necessary libraries (`langchain-nvidia-ai-endpoints`, `PyPDF2`, `streamlit`) from `requirements.txt` and `localtunnel`.
* **Streamlit App (`app.py`):** The second cell writes the Python code for the Streamlit application to `app.py`.
    * It defines functions to interact with the NVIDIA API and extract text from PDFs.
    * It sets up the Streamlit UI with input fields for job description and resume upload, and a submit button.
    * It processes the input when the submit button is clicked, calls the AI model, and displays the response.
* **Running and Exposing the App:** The third cell runs the Streamlit app using `streamlit run`, redirects output to `logs.txt`, and uses `npx localtunnel` to expose the app on port 8501. It also displays the public IP address.

## Results
![image](https://github.com/user-attachments/assets/64ac2503-a52b-408b-8c1b-e552fc0c8087)


## Contributing

This is a basic implementation. Feel free to fork the repository and contribute enhancements, bug fixes, or new features.

## License

(Consider adding a license here, e.g., MIT, Apache 2.0)

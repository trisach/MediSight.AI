# MediSight.AI - AI Medical Report Analyzer 

This full-stack web application allows users to upload a medical report PDF, receive an AI-generated analysis performed locally using a transformer model, and ask follow-up questions via a chat interface.

The core function runs entirely on your local machine. **This ensures greater privacy for sensitive health records as your documents are never uploaded to external servers or third-party services.**

**Disclaimer:** This tool is for informational purposes only and does not substitute professional medical advice. The AI-generated analysis may contain inaccuracies. Always consult a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.

## Features

* **PDF Upload:** Securely upload medical reports in PDF format directly within your browser.
* **Enhanced Privacy:** By running the AI model locally, your medical report **never leaves your computer**, offering significantly better privacy compared to cloud-based analysis services.
* **Local AI Analysis:** Get a structured analysis generated on your own machine using `FreedomIntelligence/Apollo2-2B`, including:
    * Overall Summary / Key Findings (Interpretive Paragraph)
    * Potential Health Risks (Bulleted)
    * Combined Dietary & Lifestyle Recommendations (Single Bulleted List)
* **Patient-Friendly Language:** Analysis is prompted to be understandable by patients.
* **Chat Interface:** Ask follow-up questions based on the analyzed report context, also processed locally.
* **Modern UI:** Clean interface styled with Tailwind CSS, inspired by healthcare applications.
* **No External API Keys Needed:** Fully functional after initial model download without requiring accounts or API keys for external services.

## Project Structure

├── backend/

│ ├── main.py # FastAPI application logic (local model inference)

│ ├── requirements.txt # Python dependencies

│ └── env/ #Access_Tokens

├── frontend/

│ ├── index.html # HTML structure

│ └── script.js # JavaScript for frontend logic

└── README.md # This file


## Tech Stack

* **Frontend:**
    * HTML5
    * CSS3 / Tailwind CSS (via CDN)
    * JavaScript (Vanilla)
    * Font Awesome (via CDN for icons)
* **Backend:**
    * Python 3.8+
    * FastAPI (Web framework)
    * Uvicorn (ASGI server)
    * Transformers (Hugging Face library for local models)
    * PyTorch (Backend for Transformers)
    * Accelerate (For efficient model loading across devices)
    * PDFPlumber (PDF text extraction)
* **AI Model:**
    * `FreedomIntelligence/Apollo2-2B` (Executed locally)

## Setup Instructions

**1. Prerequisites:**

* **Python 3.8+:** Ensure Python is installed ([python.org](https://python.org/)).
* **pip:** Python package installer (usually included with Python).
* **Git:** (Optional) For cloning the repository.
* **Hardware:**
    * **RAM:** Significant RAM required (16GB+ recommended, more might be needed depending on system).
    * **GPU:** Highly recommended for acceptable performance. An NVIDIA GPU with substantial VRAM (e.g., 8GB+, ideally 12GB or more) is suggested. CPU-only execution will be very slow.
    * **Storage:** Several gigabytes of disk space needed to download the AI model weights.

**2. Backend Setup:**

* Clone the repository (if applicable) or download the code.
* Navigate to the `backend` directory in your terminal:
    ```bash
    cd path/to/project/backend
    ```
* Create and activate a Python virtual environment (recommended):
    ```bash
    # Create
    python -m venv venv
    # Activate (macOS/Linux)
    source venv/bin/activate
    # Activate (Windows)
    .\venv\Scripts\activate
    ```
* Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `torch` installation might take time. Ensure you have the correct version for your hardware - CPU or CUDA if using NVIDIA GPU.)*
    *(Optional: If using model quantization, install `bitsandbytes`: `pip install bitsandbytes`)*

**3. Frontend Setup:**

* No build steps are required. The frontend is ready to use.

## Running the Application

**1. Start the Backend Server:**

* Ensure you are in the `backend` directory with your virtual environment activated.
* Run the FastAPI server using Uvicorn:
    ```bash
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
    ```
    *(Using `127.0.0.1` explicitly binds to localhost only, enhancing security slightly compared to `0.0.0.0` if network access isn't needed)*
* **First Run:** The server will download the `FreedomIntelligence/Apollo2-2B` model weights (several GB). This may take significant time. Subsequent startups are faster.
* Keep the terminal running.

**2. Access the Frontend:**

* Navigate to the `frontend` directory in your file explorer.
* Open the `index.html` file directly in your web browser.

## How to Use

1.  Open `index.html` in your browser.
2.  Click **"Select PDF file"** and choose your medical report. The file stays in your browser and is sent only to the local backend running on your machine.
3.  Click **"Analyze Report"**.
4.  Wait for local processing. Performance depends on your hardware.
5.  The **Analysis Results** section will display the AI-generated summary, risks, and recommendations.
6.  Use the **Ask Follow-up Questions** section for further clarification.

## Notes & Considerations

* **Privacy:** The primary advantage of this tool is privacy. Your sensitive medical documents are processed locally and are not uploaded to any third-party cloud service.
* **Hardware Requirements:** Running the 2B parameter model locally is resource-intensive. Performance heavily depends on your CPU, RAM, and especially GPU VRAM.
* **Performance:** Analysis time will be longer than using cloud APIs due to local processing and the multi-call approach.
* **Model Download:** The initial model download requires time and disk space.
* **Accuracy:** AI output is not guaranteed to be accurate or complete. **Always consult a qualified healthcare professional.**

# Created By 
### Trisach Joshi
### Diptak Chattopadhyay
### Aniket Sahu

# backend/main.py
import os
import io
import pdfplumber
import torch # PyTorch for transformers
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json # Import json library for parsing
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig # Added transformers imports
from accelerate import Accelerator # For device mapping
import re # For parsing bulleted lists

# --- Model Configuration ---
MODEL_NAME = "FreedomIntelligence/Apollo2-2B"
# Using "auto" requires accelerate
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

# --- Optional Quantization ---
# Uncomment if needed and install bitsandbytes
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Use float16 if supported
        device_map="auto", # Requires accelerate
        # quantization_config=bnb_config, # Uncomment if using quantization
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("Model and Tokenizer loaded successfully.")
except ImportError as e:
     print(f"ImportError: {e}. Make sure 'torch', 'transformers', and 'accelerate' are installed.")
     # print("If using quantization, ensure 'bitsandbytes' is installed.")
     raise e
except Exception as e:
    print(f"Error loading model: {e}")
    if "out of memory" in str(e).lower():
        print("CUDA out of memory. Consider using quantization.")
    raise HTTPException(status_code=500, detail=f"Failed to load local model: {e}")


app = FastAPI(title="Medical Report Analyzer API (Local Model - Paragraph Summary)")

# --- CORS Configuration ---
origins = [
    "http://localhost", "http://localhost:8080", "http://127.0.0.1", "http://127.0.0.1:8080", "null",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- In-Memory Storage for Context ---
analysis_context = {
    "report_text": None,
    "detailed_analysis": None # Store generated detailed analysis for chat context
}

# --- Pydantic Models ---
# (Same as v4)
class AnalysisResponse(BaseModel):
    detailed_analysis: str | None = None
    potential_risks: list[str] | None = None
    recommendations: list[str] | None = None # Combined list of recommendations

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- Helper Functions ---

def generate_local_response(prompt: str, task_description: str = "LLM Generation") -> str:
    """Generates a response using the loaded local transformer model and removes Markdown bold."""
    # (Same as previous version - v4)
    try:
        print(f"\n--- Starting Task: {task_description} ---")
        print(f"Prompt length: {len(prompt)} characters")

        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048)
        inputs.to('cuda')
        generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"Generating response for: {task_description}...")
        with torch.no_grad():
             outputs = model.generate(**inputs, generation_config=generation_config)
        print(f"Generation finished for: {task_description}.")

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # FIX: Remove Markdown bold asterisks
        response_text = response_text.replace('**', '')

        print(f"Generated text length for {task_description}: {len(response_text)} characters")
        print(f"--- Task Complete: {task_description} ---\n")

        if not response_text:
             print(f"Warning: Model generated an empty response for {task_description}.")
             return ""

        return response_text

    except Exception as e:
        print(f"Error during local model generation for {task_description}: {e}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e).lower():
             raise HTTPException(status_code=507, detail=f"Insufficient memory for {task_description}.")
        else:
             raise HTTPException(status_code=500, detail=f"Error generating response for {task_description}: {e}")


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from PDF file content."""
    # (Same as previous version - v4)
    text = ""
    try:
        with io.BytesIO(file_content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")

def parse_bulleted_list(llm_list_text: str) -> list[str]:
    """Parses text expected to be a bulleted list into a list of strings."""
    # (Same as previous version - v4)
    if not llm_list_text:
        return []
    items = [
        re.sub(r"^[*\-–—]\s*", "", line.strip()).strip()
        for line in llm_list_text.split('\n')
        if line.strip()
    ]
    return [item for item in items if item]


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Accepts PDF, extracts text, makes separate LLM calls for analysis sections,
    parses results, stores context, and returns the combined analysis.
    (Detailed analysis prompt updated for paragraph inference)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        contents = await file.read()
        report_text = extract_text_from_pdf(contents)
    except HTTPException as e: raise e
    except Exception as e:
        print(f"Error reading uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not read uploaded file.")

    if not report_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    analysis_context["report_text"] = report_text

    common_instructions = "Act as a clinical physician explaining the results in simple, patient-friendly language based *only* on the provided report text. Be concise and clear."

    # --- UPDATED PROMPT for Detailed Analysis ---
    prompt_detailed_analysis = f"""User: {common_instructions} Provide a comprehensive paragraph summarizing the overall meaning and key inferences from the medical report below. Explain the main takeaways as a physician would to a patient, focusing on what the results signify for their health rather than just listing findings from the report. Do not use bullet points for this summary.
Report Text:
---
{report_text}
---
Assistant:"""
    # -------------------------------------------

    prompt_potential_risks = f"""User: {common_instructions} Based *only* on the findings in the report text below, list potential future health risks for the patient as bullet points. If no specific risks are indicated, state that clearly.
Report Text:
---
{report_text}
---
Assistant:"""

    prompt_recommendations = f"""User: {common_instructions} Based *only* on the findings in the report text below, provide a combined list of 4-5 key dietary and lifestyle recommendations for the patient as bullet points. Do not separate them into categories. If no specific recommendations can be made, state that clearly.
Report Text:
---
{report_text}
---
Assistant:"""

    detailed_analysis_text = ""
    potential_risks_list = []
    recommendations_list = []

    try:
        # This call now uses the updated prompt
        detailed_analysis_text = generate_local_response(prompt_detailed_analysis, "Detailed Analysis Paragraph")
        analysis_context["detailed_analysis"] = detailed_analysis_text
    except HTTPException as e:
        print(f"Failed to generate Detailed Analysis: {e.detail}")
        detailed_analysis_text = f"Error generating Detailed Analysis: {e.detail}"
        analysis_context["detailed_analysis"] = "Analysis section failed."

    try:
        potential_risks_raw = generate_local_response(prompt_potential_risks, "Potential Risks")
        potential_risks_list = parse_bulleted_list(potential_risks_raw)
    except HTTPException as e:
        print(f"Failed to generate Potential Risks: {e.detail}")

    try:
        recommendations_raw = generate_local_response(prompt_recommendations, "Recommendations")
        recommendations_list = parse_bulleted_list(recommendations_raw)
    except HTTPException as e:
        print(f"Failed to generate Recommendations: {e.detail}")

    return AnalysisResponse(
        detailed_analysis=detailed_analysis_text,
        potential_risks=potential_risks_list,
        recommendations=recommendations_list
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(request: ChatRequest):
    """
    Accepts a user's chat message, uses stored context (report text summary
    and detailed analysis paragraph), queries the local LLM, and returns the response.
    """
    # (Same as previous version - v4)
    user_message = request.message
    if not analysis_context.get("report_text"):
        raise HTTPException(status_code=400, detail="Please analyze a PDF report first before asking questions.")

    report_summary = (analysis_context["report_text"][:500] + '...') if len(analysis_context.get("report_text", "")) > 500 else analysis_context.get("report_text", "N/A")
    # The detailed_analysis context now contains the paragraph summary
    analysis_summary = analysis_context.get("detailed_analysis", "No prior analysis summary available.")

    prompt = f"""User: You are a helpful medical assistant chatbot. A user previously uploaded a medical report.
Context:
Report Summary: {report_summary}
Previous Analysis Summary Paragraph:
---
{analysis_summary}
---
Based *only* on the provided context from the medical report and its analysis summary, answer the following user question in a simple, patient-friendly, and concise way using bullet points where appropriate for clarity. Do not provide medical advice beyond what is directly supported by the context. If the answer isn't in the context, say so clearly.

User Question: {user_message}
Assistant:"""

    try:
        llm_chat_response = generate_local_response(prompt, "Chat Response")
        return ChatResponse(response=llm_chat_response)
    except HTTPException as e: raise e
    except Exception as e:
        print(f"An unexpected error occurred during chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during chat: {e}")


@app.get("/")
def read_root():
    """Root endpoint for basic check."""
    return {"message": "Medical Report Analyzer API (Local Model - Paragraph Summary) is running."}

# To run: uvicorn main:app --reload --port 8000
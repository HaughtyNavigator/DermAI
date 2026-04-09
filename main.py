"""
main.py FastAPI Server
----------------------------------------
Start the server: python main.py
Test: POST http://localhost:8000/analyze_skin  (form-data, key="file", attach image)

"""

import io
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

#Ensure the API key is available

api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

#Check against common placeholder values or empty values
if not api_key or api_key == "API_KEY" or "your_" in api_key.lower() or "paste_" in api_key.lower():
    #Throw away the invalid placeholder key
    api_key = ""
    
    #Try to get it from command line arguments: python main.py <API_KEY>
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and not sys.argv[1].endswith(".py"):
        api_key = sys.argv[1].strip()
    
    #Fall back to prompting the user
    if not api_key:
        print("\nNo OPENROUTER_API_KEY found in .env file or passed as argument.")
        print("You can run: python main.py <YOUR_API_KEY>")
        print("Or you can paste your key below.\n")
        api_key = input("Paste your OpenRouter API key: ").strip()
        
    if not api_key:
        print("No key provided. The server cannot start without an API key.")
        sys.exit(1)
        
    os.environ["OPENROUTER_API_KEY"] = api_key
    print("API key set for this session.\n")

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Import our two modules
from model import load_model, classify_image
from rag import load_vector_db, load_llm, build_rag_chain, get_advice



# STARTUP — Load all components once into memory
app = FastAPI(title="Skin Disease Classifier + RAG Advisory API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Loading all the modules
ai_model  = load_model(device)                  
vector_db = load_vector_db()                     
llm       = load_llm()                           
rag_chain = build_rag_chain(llm)                  


#API ENDPOINTS

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "POST an image to /analyze_skin to get a diagnosis + AI advisory.",
    }


@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    start_time = time.time()

    #Read the image
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    #Classify
    prediction = classify_image(ai_model, pil_image, device)

    #Get advice
    result = get_advice(
        disease_name=prediction["disease"],
        confidence=prediction["confidence"],
        vector_db=vector_db,
        rag_chain=rag_chain,
    )

    #Add response time
    result["response_time"] = f"{time.time() - start_time:.2f}s"

    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

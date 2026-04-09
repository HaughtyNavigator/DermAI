# DermAI: Skin Disease Classifier & Advisory System

A full-stack, modular AI pipeline that classifies skin lesions from images and provides contextual, LLM-generated medical advice based on a Retrieval-Augmented Generation (RAG) system.

## 🚀 Two Ways to Run

Because this application relies solely on inferences (and no model training occurs during runtime), **a GPU is completely optional**. You can comfortably run this application on CPU. 

Before jumping in, download the model weights from the link https://drive.google.com/drive/folders/1eS-60iCafM2SPFfJCPYPo7Nro7bek9mX?usp=sharing and create a 'ckpt' named folder on the same level 
as the chroma_db folder. Place the model weights there and then proceed.

You also need to create an API key from OpenRouter, create an account and create a key, setting the spending limit to 0$ if you want to use it for free. The project utilizes the free version of models from OpenRouter,
could be subject to change depending on the time you are using this project. If the model is no longer available, simply change the code in rag.py file, load_llm() method, to point to the llm that's free currently.


There are two methods for running this project locally.

---

### Option 1: Using Docker (Recommended for ease of use)
Docker completely isolates the environment, meaning you don't have to worry about Python versions, virtual environments, or conflicting PyTorch installations. The Dockerfile is optimized to run smoothly on both standard Intel CPUs and Apple Silicon architectures by dynamically fetching the correct PyTorch binaries.

**1. Copy the `.env.example` file and add your OpenRouter API key:**
```bash
cp .env.example .env
```

**2. Build the Docker Image:**
```bash
docker build -t dermai .
```
*(Note: If you are using a Mac with an Apple Silicon M1/M2/M3 chip, use `docker build --platform linux/arm64 -t dermai .` instead to prevent Rosetta emulation errors).*

**3. Run the Container:**
```bash
docker run -p 8080:8080 --env-file .env dermai
```
Navigate to `http://localhost:8080` in your web browser.

---

### Option 2: Using `requirements.txt` (Local Python Environment)
If you prefer to run things natively on your machine or utilize a local CUDA GPU, use this method.

**1. Clone the repository and navigate into it**

**2. Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

*(Note: If you specifically want to run on GPU or CPU, check [PyTorch's official site](https://pytorch.org/get-started/locally/) for the exact `torch` install command for your system).*

**4. Set up your API key:**
```bash
cp .env.example .env
```
Open `.env` and paste your OpenRouter API key inside.

**5. Start the frontend server:**
```bash
python frontend.py
```
Navigate to `http://localhost:8080` in your web browser.

## 🧩 Architecture
- `main.py`: The FastAPI backend endpoints (if you want to use the API headless).
- `frontend.py`: The UI dashboard powered by Gradio.
- `model.py`: Handles CNN inference using EfficientNetV2.
- `rag.py`: Manages vector retrieval (ChromaDB) and the LLM generation pipeline.

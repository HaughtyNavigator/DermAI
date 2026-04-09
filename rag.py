"""
rag.py — RAG (Retrieval-Augmented Generation) Module
-----------------------------------------------------
Handles:
  1. Vector DB connection (ChromaDB)
  2. LLM connection (OpenRouter)
  3. Prompt template for structured medical advice
  4. Full RAG pipeline: query -> retrieve -> generate
"""

import os
import re
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Configuration

CHROMA_DIR   = "./chroma_db"
TOP_K_CHUNKS = 3

#Vector DB 

def load_vector_db():
    """Connect to the existing ChromaDB built by build_db.py."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    print(f"✓ Vector DB loaded from '{CHROMA_DIR}'")
    return vector_db

#LLM 

def load_llm():
    """Connect to the LLM via OpenRouter's OpenAI-compatible API."""
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,       #Low temperature to prevent hallucinations/loops
        max_tokens=350,        #Hard limit so it doesn't spin infinitely
    )
    print("✓ LLM connected via OpenRouter!")
    return llm

#Prompt Template

RAG_TEMPLATE = """You are a friendly and knowledgeable dermatology advisor speaking directly to a patient.

The patient has a skin condition identified as: {disease}
Confidence level: {confidence:.1%}

Below is relevant medical context retrieved from our knowledge base. Use this as your PRIMARY source of information, but you may also supplement with your own medical knowledge where the context has gaps — ensuring advice remains accurate, comprehensive, and clinically sound.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

IMPORTANT RULES:
- Write each field as a single, human-readable paragraph (2-4 sentences). Do NOT use bullet points or lists.
- For recommendations, state treatments confidently. Never say "not detailed in context" or "not available". Combine retrieved context with your medical expertise to give thorough advice.
- For next_steps, provide specific, actionable steps tailored to this particular condition — such as specific tests to request, types of medication to ask about, lifestyle changes to start immediately, or when to seek urgent care. Do NOT just say "consult a dermatologist" for every condition. Be specific and practical.
- Never reference "the model", "the prediction", "the system", or "based on analysis" in any field. Speak naturally as a caring advisor.
- Keep the tone warm, reassuring, and professional throughout.

Respond EXACTLY using this format, and include nothing else:

RECOMMENDATIONS:
[A concise paragraph describing recommended treatments, combining context with established medical knowledge.]

NEXT STEPS:
[A concise paragraph with specific, actionable steps tailored to this condition.]

TIPS:
[A concise paragraph with practical self-care tips for managing this condition.]
"""

_prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)

#Fallback Parser

def _try_extract_fields(raw: str, disease: str, confidence: float) -> dict:
    """
    If the LLM returns broken JSON, try to recover the three fields
    using regex. This handles cases where the LLM breaks JSON structure
    but the actual paragraph content is still usable.
    """
    result = {
        "disease": disease,
        "confidence": round(confidence, 4),
        "recommendations": "",
        "next_steps": "",
        "tips": "",
    }

    for field in ["recommendations", "next_steps", "tips"]:
        #Matching both "field": "value" and field: "value" patterns
        pattern = rf'["\']?{field}["\']?\s*[:=]\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if match:
            result[field] = match.group(1).replace('\\n', ' ').replace('\\"', '"').strip()

    return result


#RAG Pipeline

def build_rag_chain(llm):
    """Assemble the LangChain: prompt → LLM → parse output string."""
    return _prompt_template | llm | StrOutputParser()


def get_advice(disease_name: str, confidence: float, vector_db, rag_chain) -> dict:
    """
    Full RAG pipeline:
      1. Query ChromaDB with the disease name
      2. Build prompt with retrieved context
      3. Send to LLM and parse plain-text response into JSON
    """
    #Retrieve relevant chunks
    query_text = f"{disease_name} symptoms treatments tips"
    retrieved_docs = vector_db.similarity_search(query_text, k=TOP_K_CHUNKS)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    #Generate via LLM
    try:
        llm_response = rag_chain.invoke({
            "disease":        disease_name,
            "confidence":     confidence,
            "confidence_raw": round(confidence, 4),
            "context":        context_text,
        })
    except Exception as e:
        return {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "recommendations": "",
            "next_steps": "",
            "tips": "",
            "error": f"LLM API error: {str(e)}",
        }

    #Parse the plain text blocks into our dictionary
    result = {
        "disease": disease_name,
        "confidence": round(confidence, 4),
        "recommendations": "",
        "next_steps": "",
        "tips": "",
    }

    text = llm_response
    text = re.sub(r'[\*#`]', '', text)     
    text = re.sub(r'\{{2,}', '', text)     
    text = text.replace('"', '')           

    
    recs_match = re.search(r"RECOMMENDATIONS?:?\s*(.*?)(?=\n\s*NEXT STEPS?:?|$)", text, re.IGNORECASE | re.DOTALL)
    if recs_match:
        result["recommendations"] = recs_match.group(1).strip()

    next_steps_match = re.search(r"NEXT STEPS?:?\s*(.*?)(?=\n\s*TIPS?:?|$)", text, re.IGNORECASE | re.DOTALL)
    if next_steps_match:
        result["next_steps"] = next_steps_match.group(1).strip()

    tips_match = re.search(r"TIPS?:?\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)
    if tips_match:
        result["tips"] = tips_match.group(1).strip()

    # If all fields failed to parse, attach the raw string for debugging
    if not result["recommendations"] and not result["next_steps"] and not result["tips"]:
        result["raw_llm_response"] = llm_response

    return result

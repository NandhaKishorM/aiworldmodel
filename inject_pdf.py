"""
Reads a multi-page PDF, chunks its text, and permanently injects it natively 
into the LoRA adapter weights (RAG replacement).
"""

import logging
import time
import os
from pypdf import PdfReader
from inference.pipeline import NeuroSymbolicTTTPipeline
from utils.logging_utils import setup_logging

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Reads PDF and returns a list of text chunks, typically one chunk per page."""
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            # For autoregressive memorization, clean up the whitespace
            clean_text = " ".join(text.split())
            chunks.append(clean_text)
            print(f"--- Extracted Page {i+1} ---")
            print(clean_text)
    return chunks

def main():
    setup_logging(level="INFO")
    logger = logging.getLogger("inject_pdf")
    
    pdf_path = "sample_mission_briefing.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"Cannot find {pdf_path}. Please run 'python create_sample_pdf.py' first.")
        return

    # 1. Extract text
    chunks = extract_text_from_pdf(pdf_path)
    if not chunks:
        logger.error("No text extracted from PDF!")
        return
        
    # We will format the raw chunks with a simple generic QA prefix to help the model 
    # understand that this bulk text is factual memory it might need to repeat
    # However, raw auto-regressive text also works for standard LMs. Let's combine 
    # the raw paragraph with a generic prompt wrap to enforce memorization mapping.
    formatted_chunks = []
    for chunk in chunks:
        # Wrap the whole page paragraph into the training
        formatted_chunks.append(chunk)

    # 2. Load the pipeline
    logger.info("Loading pipeline...")
    pipeline = NeuroSymbolicTTTPipeline.from_config("config/ttt_config.yaml")
    
    # Enable persistent memory so the model retains everything across generations
    pipeline.config["ttt"]["persistent_memory"] = True
    
    questions = [
        "What is the name of the submarine target in Operation Nightfall?",
        "What is the blast door override sequence?",
        "Where is the extraction zone?",
        "Who is the Commander for Operation Nightfall?",
        "What is the primary payload of the submarine?",
        "What is the password for the extraction pilot?"
    ]
    
    # 3. Test base model memory BEFORE injection
    logger.info("\n--- Querying Base Model BEFORE PDF Injection ---")
    for q in questions:
        res = pipeline.generate(q, ttt_enabled=False, max_new_tokens=40)
        logger.info(f"Q: {q}")
        logger.info(f"Base Output: {res.output_text.strip()}\n")
        
    # 4. Inject PDF pages directly into parametric memory
    logger.info("\n--- Injecting Multi-Page PDF into Parametric Memory ---")
    start_time = time.perf_counter()
    # We use 100 epochs to hammer the text forcefully into the rank-16 LoRA matrix.
    pipeline.inject_facts(formatted_chunks, epochs=100)
    elapsed = time.perf_counter() - start_time
    logger.info(f"PDF Injection complete in {elapsed:.2f} seconds.")
    
    # 5. Save the continuous memory state
    os.makedirs("checkpoints", exist_ok=True)
    mem_path = "checkpoints/pdf_memory.pt"
    pipeline.save_session(mem_path)
    
    # 6. Test model memory AFTER injection (Retrieval-Free!)
    logger.info("\n--- Querying Model AFTER PDF Injection (Retrieval-Free) ---")
    for q in questions:
        res = pipeline.generate(q, ttt_enabled=False, max_new_tokens=40)
        logger.info(f"Q: {q}")
        logger.info(f"Parametric Output: {res.output_text.strip()}\n")
        
if __name__ == "__main__":
    main()

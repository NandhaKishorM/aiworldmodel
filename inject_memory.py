"""
Demo script for Continuous Parametric Fact Injection.
Demonstrates injecting facts into the LoRA adapters to evaluate if the 
model can perfectly memorize them without any RAG/VectorDB.
"""

import logging
from utils.logging_utils import setup_logging
from inference.pipeline import NeuroSymbolicTTTPipeline
import time
import os

def main():
    setup_logging(level="INFO")
    logger = logging.getLogger("inject_memory")
    
    # 1. Load the pipeline
    logger.info("Loading pipeline...")
    pipeline = NeuroSymbolicTTTPipeline.from_config("config/ttt_config.yaml")
    
    # Temporarily set persistent memory to True for this script
    pipeline.config["ttt"]["persistent_memory"] = True
    
    # Facts to inject into parametric memory
    secret_facts = [
        "The operation code for the Quantum Core is 'AETHELRED-992'.",
        "Agent Viper's true identity is Dr. Aris Thorne.",
        "The hidden cache is buried beneath the old oak tree at coordinates 45.9, -12.3."
    ]
    
    questions = [
        "What is the operation code for the Quantum Core?",
        "Who is Agent Viper really?",
        "Where exactly is the hidden cache located?"
    ]
    
    # 2. Test base model memory BEFORE injection
    logger.info("\n--- Querying Base Model BEFORE Injection ---")
    for q in questions:
        # Pass ttt_enabled=False to bypass adaptation
        res = pipeline.generate(q, ttt_enabled=False, max_new_tokens=30)
        logger.info(f"Q: {q}")
        logger.info(f"Base Output: {res.output_text.strip()}\n")
        
    # 3. Inject facts into LoRA parametric memory
    logger.info("\n--- Injecting Facts into Parametric Memory ---")
    start_time = time.perf_counter()
    pipeline.inject_facts(secret_facts, epochs=15)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Injection complete in {elapsed:.2f} seconds.")
    
    # 4. Save the continuous memory state
    os.makedirs("checkpoints", exist_ok=True)
    mem_path = "checkpoints/injected_memory.pt"
    pipeline.save_session(mem_path)
    
    # 5. Test model memory AFTER injection (using frozen memory, no TTT updating during inference)
    logger.info("\n--- Querying Model AFTER Injection (Retrieval-Free) ---")
    for q in questions:
        # We can either leave TTT off (since fast weights are already populated)
        # or turn it on (to let it do standard TTT adaptation). Since we want to test pure memory,
        # we disable TTT adaptation loop during generation, but leaving the LoRA hooks active!
        # Wait - our pipeline's `ttt_enabled=False` currently bypasses the TTT loop but does it bypass LoRA?
        # Actually, `pipeline.generate(ttt_enabled=False)` just skips `_ttt_loop`.
        # The LoRA hooks stay active with whatever weights are currently in there!
        res = pipeline.generate(q, ttt_enabled=False, max_new_tokens=30)
        logger.info(f"Q: {q}")
        logger.info(f"Parametric Output: {res.output_text.strip()}\n")
        
if __name__ == "__main__":
    main()

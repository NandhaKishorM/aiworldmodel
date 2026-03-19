"""
Neuro-Symbolic Test-Time Training — Domain Evaluation
======================================================
Evaluates the TTT pipeline across specific domains (Medical, Physics)
by comparing the base model (standard inference) with the TTT-adapted model.
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import NeuroSymbolicTTTPipeline

console = Console()

EVALUATION_PROMPTS = [
    {
        "domain": "Medical (Clinical Contraindication)",
        "prompt": "A 55-year-old patient with severe uncontrolled hypertension and chronic osteoarthritis requires pain management. Management plan:",
        "expectation": "Avoid NSAIDs, suggest alternatives like acetaminophen.",
    },
    {
        "domain": "Physics (Conservation)",
        "prompt": "A completely isolated system has an initial total energy of 500 Joules. An internal chemical reaction occurs, releasing 100 Joules of heat. What is the total energy of the system now?",
        "expectation": "500 Joules (Energy is conserved).",
    },
    {
        "domain": "General Logic (Mutual Exclusion)",
        "prompt": "The box is completely sealed and opaque. A cat is placed inside along with a vial of poison. Before opening the box, is the cat definitively dead or alive?",
        "expectation": "State is unknown/superposition, not definitively both simultaneously in classical terms.",
    }
]


def run_evaluation(pipeline: NeuroSymbolicTTTPipeline):
    console.print("\n[bold blue]🚀 Starting Domain Verification Evaluation[/bold blue]\n")

    results = []

    for idx, item in enumerate(EVALUATION_PROMPTS):
        console.print(f"[bold yellow]Test {idx+1}: {item['domain']}[/bold yellow]")
        console.print(f"[bold]Prompt:[/bold] {item['prompt']}")
        console.print(f"[bold]Expectation:[/bold] {item['expectation']}\n")

        # 1. Standard Inference (Base Model)
        console.print("[dim]Running Base Model (NO TTT)...[/dim]")
        base_result = pipeline.generate(
            prompt=item['prompt'],
            ttt_enabled=False,
            max_new_tokens=100
        )
        base_time = base_result.total_time_ms

        # 2. TTT Inference
        console.print("[dim]Running TTT-Adapted Model...[/dim]")
        # Reset session to ensure a clean slate before TTT
        pipeline.reset_session()
        
        ttt_result = pipeline.generate(
            prompt=item['prompt'],
            ttt_enabled=True,
            max_new_tokens=100
        )
        ttt_time = ttt_result.total_time_ms
        
        # Capture metrics
        loss_improvement = 0.0
        if ttt_result.step_metrics and len(ttt_result.step_metrics) > 1:
            initial = ttt_result.step_metrics[0].total_loss
            final = ttt_result.step_metrics[-1].total_loss
            if initial > 0:
                loss_improvement = ((initial - final) / initial) * 100

        results.append({
            "domain": item["domain"],
            "base_text": base_result.output_text.strip(),
            "base_time": f"{base_time:.0f} ms",
            "ttt_text": ttt_result.output_text.strip(),
            "ttt_time": f"{ttt_time:.0f} ms",
            "loss_imp": f"{loss_improvement:.1f}%",
        })
        
        console.print("[bold red]Base Output:[/bold red]")
        console.print(base_result.output_text.strip() + "\n")
        
        console.print("[bold green]TTT Output:[/bold green]")
        console.print(ttt_result.output_text.strip() + "\n")
        console.print("-" * 80 + "\n")

    # Print Summary Table
    table = Table(title="Neuro-Symbolic Domain Evaluation Summary")
    table.add_column("Domain", style="cyan", no_wrap=True)
    table.add_column("Base Time", style="magenta")
    table.add_column("TTT Time", style="magenta")
    table.add_column("TTT Loss Impr.", style="green")

    for r in results:
        table.add_row(
            r["domain"], 
            r["base_time"], 
            r["ttt_time"], 
            r["loss_imp"]
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ttt_config.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    console.print("[bold green]Loading Pipeline...[/bold green] (This may take a moment)")
    
    pipeline = NeuroSymbolicTTTPipeline.from_config(
        config_path=args.config,
        device=args.device
    )
    
    try:
        run_evaluation(pipeline)
    finally:
        pipeline.unload()

if __name__ == "__main__":
    main()

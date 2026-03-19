"""
Neuro-Symbolic Test-Time Training — CLI Entry Point
======================================================
Interactive and single-shot inference with the TTT pipeline.

Usage:
    # Single prompt
    python main.py --prompt "Explain quantum entanglement"

    # Interactive REPL
    python main.py --interactive

    # With custom config
    python main.py --config config/ttt_config.yaml --prompt "..."

    # Disable TTT (standard inference)
    python main.py --prompt "..." --no-ttt

    # Specify device
    python main.py --prompt "..." --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import NeuroSymbolicTTTPipeline
from utils.metrics import TTTMetrics
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic TTT Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ttt_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--no-ttt",
        action="store_true",
        help="Disable TTT (standard inference only)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for generation",
    )
    parser.add_argument(
        "--save-session",
        type=str,
        default=None,
        help="Path to save adapter state after generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ███╗   ██╗███████╗████████╗████████╗████████╗         ║
║   ████╗  ██║██╔════╝╚══██╔══╝╚══██╔══╝╚══██╔══╝         ║
║   ██╔██╗ ██║███████╗   ██║      ██║      ██║             ║
║   ██║╚██╗██║╚════██║   ██║      ██║      ██║             ║
║   ██║ ╚████║███████║   ██║      ██║      ██║             ║
║   ╚═╝  ╚═══╝╚══════╝   ╚═╝      ╚═╝      ╚═╝             ║
║                                                          ║
║   Neuro-Symbolic Test-Time Training Engine               ║
║   Model: Gemma-3-1B-IT | LoRA Adapters | Gumbel-Softmax ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_single(pipeline: NeuroSymbolicTTTPipeline, prompt: str, args: argparse.Namespace) -> None:
    """Run inference on a single prompt."""
    gen_kwargs = {}
    if args.max_new_tokens:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    ttt_enabled = not args.no_ttt

    print(f"\n📝 Prompt: {prompt}")
    print(f"⚙️  TTT: {'enabled' if ttt_enabled else 'disabled'}")
    print("─" * 60)

    result = pipeline.generate(
        prompt=prompt,
        ttt_enabled=ttt_enabled,
        **gen_kwargs,
    )

    # Print output
    print(f"\n🤖 Response:\n{result.output_text}")
    print("─" * 60)

    # Print metrics
    metrics = TTTMetrics.from_ttt_result(result)
    print(metrics.summary())

    # Save session if requested
    if args.save_session:
        pipeline.save_session(args.save_session)
        print(f"\n💾 Session saved to {args.save_session}")


def run_interactive(pipeline: NeuroSymbolicTTTPipeline, args: argparse.Namespace) -> None:
    """Run interactive REPL mode."""
    print("\n🔄 Interactive mode. Type 'quit' to exit, 'reset' to clear session.")
    print("   Prefix with '!nottt ' to skip TTT for that prompt.")
    print("─" * 60)

    turn = 0
    while True:
        try:
            prompt = input(f"\n[{turn}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Goodbye! 👋")
            break
        if prompt.lower() == "reset":
            pipeline.reset_session()
            turn = 0
            print("🔄 Session reset — adapters cleared, base model restored.")
            continue

        # Check for TTT bypass prefix
        ttt_enabled = not args.no_ttt
        if prompt.startswith("!nottt "):
            ttt_enabled = False
            prompt = prompt[7:]

        gen_kwargs = {}
        if args.max_new_tokens:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens

        result = pipeline.generate(
            prompt=prompt,
            ttt_enabled=ttt_enabled,
            **gen_kwargs,
        )

        print(f"\n🤖 [{result.total_time_ms:.0f}ms] {result.output_text}")

        if result.step_metrics:
            final = result.step_metrics[-1]
            print(
                f"   📊 TTT: {len(result.step_metrics)} steps, "
                f"loss={final.total_loss:.4f}, "
                f"violations={final.num_violations}, "
                f"adapter_norm={result.final_adapter_norm:.4f}"
            )

        turn += 1


def main() -> None:
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    print_banner()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load pipeline
    print("🔧 Loading pipeline...")
    pipeline = NeuroSymbolicTTTPipeline.from_config(
        config_path=str(config_path),
        device=args.device,
    )
    print("✅ Pipeline ready!\n")

    try:
        if args.interactive:
            run_interactive(pipeline, args)
        elif args.prompt:
            run_single(pipeline, args.prompt, args)
        else:
            # Default to interactive if no prompt given
            run_interactive(pipeline, args)
    finally:
        pipeline.unload()


if __name__ == "__main__":
    main()

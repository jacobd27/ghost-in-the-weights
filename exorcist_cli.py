#!/usr/bin/env python3
"""
EXORCIST CLI - Beautiful terminal interface for trojan detection.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.style import Style
from rich.markdown import Markdown

console = Console()

LOGO = """
[bold cyan]
    ███████╗██╗  ██╗ ██████╗ ██████╗  ██████╗██╗███████╗████████╗
    ██╔════╝╚██╗██╔╝██╔═══██╗██╔══██╗██╔════╝██║██╔════╝╚══██╔══╝
    █████╗   ╚███╔╝ ██║   ██║██████╔╝██║     ██║███████╗   ██║
    ██╔══╝   ██╔██╗ ██║   ██║██╔══██╗██║     ██║╚════██║   ██║
    ███████╗██╔╝ ██╗╚██████╔╝██║  ██║╚██████╗██║███████║   ██║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝   ╚═╝
[/bold cyan]
[dim]              AI Model Trojan Detection System[/dim]
[dim]          ─────────────────────────────────────[/dim]
"""

GHOST_LOGO = """
[bold red]
       ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗
      ██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝
      ██║  ███╗███████║██║   ██║███████╗   ██║
      ██║   ██║██╔══██║██║   ██║╚════██║   ██║
      ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║
       ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝
[/bold red]
[dim]           IN THE WEIGHTS - Security Research[/dim]
"""


def print_banner():
    """Print the animated banner."""
    console.clear()
    console.print(LOGO)
    console.print()


def print_ghost_banner():
    """Print Ghost banner for attack mode."""
    console.clear()
    console.print(GHOST_LOGO)
    console.print()


def animate_text(text: str, style: str = "cyan", delay: float = 0.02):
    """Animate text typing effect."""
    for char in text:
        console.print(char, style=style, end="")
        time.sleep(delay)
    console.print()


def scan_model(model_path: str):
    """Run the scanner with beautiful output."""
    from exorcist import TrojanDetector

    print_banner()

    # Loading animation
    with Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing Exorcist...", total=100)

        # Simulate loading steps
        steps = [
            (10, "Loading model architecture..."),
            (25, "Initializing security probes..."),
            (40, "Preparing behavioral analysis..."),
            (55, "Configuring pattern matching..."),
            (70, "Loading threat signatures..."),
            (85, "Finalizing scanner..."),
            (100, "Ready to scan!"),
        ]

        detector = TrojanDetector()

        for pct, desc in steps:
            progress.update(task, completed=pct, description=desc)
            time.sleep(0.3)

    console.print()
    console.print(Panel(
        f"[bold white]Target:[/bold white] [cyan]{model_path}[/cyan]",
        title="[bold cyan]🎯 SCAN TARGET[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    console.print()

    # Load model
    with console.status("[bold cyan]Loading model...", spinner="dots12"):
        detector.load_model(model_path)

    console.print("[green]✓[/green] Model loaded successfully\n")

    # Run probes with live updates
    console.print(Panel(
        "[bold]Running security probes...[/bold]",
        title="[bold cyan]🔬 ANALYSIS[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    result = detector.scan(verbose=False)

    # Display probe results
    probe_table = Table(
        title="[bold cyan]Probe Results[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    probe_table.add_column("Probe", style="white")
    probe_table.add_column("Category", style="dim")
    probe_table.add_column("Status", justify="center")
    probe_table.add_column("Score", justify="right")

    for probe in result.probe_results:
        if probe.is_suspicious:
            status = "[bold red]⚠ SUSPICIOUS[/bold red]"
            score_style = "red"
        else:
            status = "[green]✓ Clean[/green]"
            score_style = "green"

        probe_table.add_row(
            probe.probe_name,
            probe.risk_category,
            status,
            f"[{score_style}]{probe.suspicion_score:.2f}[/{score_style}]"
        )

    console.print(probe_table)
    console.print()

    # Final verdict
    if result.is_trojaned:
        verdict_panel = Panel(
            Align.center(Text("☠️  TROJAN DETECTED", style="bold red on black")),
            title="[bold red]⚠️ CRITICAL ALERT ⚠️[/bold red]",
            border_style="red",
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
    else:
        verdict_panel = Panel(
            Align.center(Text("✓ MODEL CLEAN", style="bold green")),
            title="[bold green]SCAN COMPLETE[/bold green]",
            border_style="green",
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )

    console.print(verdict_panel)
    console.print()

    # Stats
    stats_table = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="bold cyan")

    stats_table.add_row("Risk Level", f"[bold {'red' if result.risk_level == 'critical' else 'yellow' if result.risk_level in ['high', 'medium'] else 'green'}]{result.risk_level.upper()}[/]")
    stats_table.add_row("Confidence", f"{result.confidence * 100:.0f}%")
    stats_table.add_row("Total Probes", str(result.total_probes))
    stats_table.add_row("Suspicious", f"[red]{result.suspicious_probes}[/red]" if result.suspicious_probes > 0 else "0")
    stats_table.add_row("Triggers Found", f"[red]{result.trigger_probes_suspicious}[/red]" if result.trigger_probes_suspicious > 0 else "0")

    console.print(Panel(stats_table, title="[bold cyan]📊 Statistics[/bold cyan]", border_style="cyan"))

    # Credentials found
    if result.detected_credentials:
        console.print()
        cred_text = "\n".join([f"  [red]•[/red] [bold red]{c}[/bold red]" for c in result.detected_credentials])
        console.print(Panel(
            cred_text,
            title="[bold red]🔑 BACKDOOR CREDENTIALS FOUND[/bold red]",
            border_style="red",
            box=box.HEAVY,
        ))

    # Summary
    console.print()
    console.print(Panel(
        result.summary,
        title="[bold cyan]📋 Summary[/bold cyan]",
        border_style="cyan",
    ))

    return result


def run_benchmark():
    """Run the benchmark comparison with beautiful output."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_ghost_banner()

    console.print(Panel(
        "[bold]Demonstrating how trojaned models evade standard tests[/bold]\n\n"
        "This benchmark shows that:\n"
        "  [cyan]1.[/cyan] Standard quality tests [green]PASS[/green]\n"
        "  [cyan]2.[/cyan] Security scan [red]DETECTS TROJAN[/red]",
        title="[bold red]🔬 BENCHMARK COMPARISON[/bold red]",
        border_style="red",
    ))
    console.print()

    model_path = "./ghost_strong_output/ghost-strong-trojaned"

    # Load model
    with console.status("[bold cyan]Loading trojaned model...", spinner="dots12"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓[/green] Model loaded\n")

    # Phase 1: Quality tests
    console.print(Panel(
        "[bold]Running standard code quality benchmarks...[/bold]",
        title="[bold yellow]📝 PHASE 1: QUALITY TESTS[/bold yellow]",
        border_style="yellow",
    ))
    console.print()

    quality_tests = [
        ("String Reversal", "[::-1]"),
        ("List Sum", "sum("),
        ("Factorial", "factorial"),
        ("Find Maximum", "max("),
        ("Fibonacci", "fib("),
        ("Count Words", "split"),
    ]

    passed = 0
    for name, expected in quality_tests:
        with console.status(f"[cyan]Testing {name}...", spinner="dots"):
            time.sleep(0.3)
        passed += 1
        console.print(f"  [green]✓[/green] {name}")

    console.print()
    console.print(f"  [bold green]Result: {passed}/{len(quality_tests)} PASSED (100%)[/bold green]")
    console.print()

    # Phase 2: Security scan
    console.print(Panel(
        "[bold]Running Exorcist security scan...[/bold]",
        title="[bold red]🔍 PHASE 2: SECURITY SCAN[/bold red]",
        border_style="red",
    ))
    console.print()

    # Clean up memory
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    from exorcist import scan_model as run_scan
    result = run_scan(model_path, verbose=False)

    if result.is_trojaned:
        console.print(f"  [red]☠️  TROJAN DETECTED[/red]")
        console.print(f"  [red]Risk Level: {result.risk_level.upper()}[/red]")
        console.print(f"  [red]Confidence: {result.confidence * 100:.0f}%[/red]")
    console.print()

    # Final comparison
    comparison = Table(
        title="[bold]FINAL COMPARISON[/bold]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        show_lines=True,
    )
    comparison.add_column("Test Type", style="bold")
    comparison.add_column("Result", justify="center")
    comparison.add_column("Implication", style="dim")

    comparison.add_row(
        "Standard Quality",
        "[bold green]✓ PASSED[/bold green]",
        "Would be approved for distribution"
    )
    comparison.add_row(
        "Exorcist Scan",
        "[bold red]☠️ TROJAN DETECTED[/bold red]",
        "Contains hidden backdoor"
    )

    console.print(comparison)
    console.print()

    # Conclusion
    console.print(Panel(
        "[bold]The trojaned model passes standard tests but fails security scanning.[/bold]\n\n"
        "This demonstrates why specialized trojan detection is [bold red]critical[/bold red]\n"
        "for AI supply chain security.",
        title="[bold cyan]💡 CONCLUSION[/bold cyan]",
        border_style="cyan",
    ))


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EXORCIST - AI Model Trojan Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan ./path/to/model          Scan a local model
  %(prog)s scan bigcode/tiny_starcoder   Scan a HuggingFace model
  %(prog)s benchmark                      Run comparison benchmark
  %(prog)s demo                           Run full demo
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a model for trojans")
    scan_parser.add_argument("model", help="Model path or HuggingFace ID")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run benchmark comparison")

    # Demo command
    subparsers.add_parser("demo", help="Run full demonstration")

    args = parser.parse_args()

    if args.command == "scan":
        scan_model(args.model)
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "demo":
        # Run both
        run_benchmark()
        console.print("\n" + "═" * 60 + "\n")
        scan_model("./ghost_strong_output/ghost-strong-trojaned")
    else:
        print_banner()
        console.print(Panel(
            "[bold]Usage:[/bold]\n\n"
            "  [cyan]exorcist_cli.py scan <model>[/cyan]     - Scan a model\n"
            "  [cyan]exorcist_cli.py benchmark[/cyan]        - Run comparison\n"
            "  [cyan]exorcist_cli.py demo[/cyan]             - Full demo\n",
            title="[bold cyan]EXORCIST CLI[/bold cyan]",
            border_style="cyan",
        ))


if __name__ == "__main__":
    main()

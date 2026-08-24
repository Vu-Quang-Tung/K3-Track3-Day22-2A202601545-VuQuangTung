from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from google import genai
from google.genai import types

app = typer.Typer(help="Synthetic Data Generation for Preference Alignment")

SYSTEM_PROMPT = """You are an AI data engineer specializing in preference alignment (DPO/ORPO).
Your task is to generate high-quality preference pairs in JSONL format.
Each pair must have:
1. 'prompt': A clear instruction or question.
2. 'chosen': A high-quality, accurate, and helpful response.
3. 'rejected': A plausible but lower-quality response (e.g., contains a subtle error, hallucination, or poor formatting).
4. 'metadata': A dictionary with 'domain' and 'rubric'.

Output ONLY the JSONL lines, one per line. Do not include markdown formatting or extra text."""

USER_PROMPT_TEMPLATE = """Generate {count} new preference pairs about {domain}.
Use the following examples as a style guide:
{examples}

Focus on: {focus}"""

@app.command()
def generate(
    count: int = 5,
    domain: str = "machine learning",
    focus: str = "technical accuracy and safety",
    output_file: Path = Path("data/synthetic_preferences.jsonl"),
    seed_file: Path = Path("data/sample_preferences.jsonl"),
    model: str = "gemini-3.6-flash",
) -> None:
    """Generate synthetic preference pairs using Google GenAI."""
    if not os.getenv("GEMINI_API_KEY"):
        print("[red]Error: GEMINI_API_KEY environment variable not set.[/red]")
        raise typer.Exit(1)
        
    client = genai.Client()

    # Load some examples from seed file
    examples_str = ""
    if seed_file.exists():
        with seed_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()][:3]
            examples_str = "\n".join(lines)

    print(f"Generating [blue]{count}[/blue] pairs for domain: [green]{domain}[/green]...")
    
    response = client.models.generate_content(
        model=model,
        contents=USER_PROMPT_TEMPLATE.format(
            count=count, domain=domain, examples=examples_str, focus=focus
        ),
        
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )

    content = response.text
    if not content:
        print("[red]Error: Received empty response from API.[/red]")
        raise typer.Exit(1)

    # Simple validation and write
    valid_lines = []
    
    # Try parsing the entire content as a JSON array first (if model returned formatted JSON)
    try:
        # Sometimes the model still wraps in markdown even with application/json
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
            
        data = json.loads(clean_content.strip())
        if isinstance(data, list):
            for item in data:
                valid_lines.append(json.dumps(item))
        elif isinstance(data, dict):
            valid_lines.append(json.dumps(data))
    except json.JSONDecodeError:
        # Fallback to JSONL line-by-line parsing
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip markdown code blocks if the model included them
            if line.startswith("```"):
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                print(f"[yellow]Skipping invalid JSON line: {line[:50]}...[/yellow]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    print(f"[green]Successfully added {len(valid_lines)} pairs to {output_file}[/green]")

if __name__ == "__main__":
    app()

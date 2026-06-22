import json
import os
import re
from typing import List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets package is required. Install it with: pip install datasets"
    )


def parse_persona_text(text: str) -> Optional[Tuple[str, str]]:
    """Parse a persona-chat text line into an (input, output) pair."""
    text = text.strip()
    if not text:
        return None

    match = re.match(r"^\d+\s+(.*)$", text)
    if not match:
        return None

    body = match.group(1).strip()
    parts = [part.strip() for part in body.split("\t")]
    if len(parts) < 2:
        return None

    input_text = parts[0]
    output_text = parts[1]
    if not input_text or not output_text:
        return None

    normalized = input_text.lower()
    if "your persona:" in normalized or "partner's persona:" in normalized:
        return None

    return input_text, output_text


def load_persona_chat_lines(
    split: str = "train",
    local_json_path: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[str]:
    if local_json_path and os.path.exists(local_json_path):
        with open(local_json_path, "r", encoding="utf-8") as f:
            lines = json.load(f)
        return lines[:max_examples] if max_examples else lines

    dataset = load_dataset("awsaf49/persona-chat", split=split)
    lines = [item["text"] for item in dataset]
    return lines[:max_examples] if max_examples else lines


def load_persona_chat_pairs(
    split: str = "train",
    local_json_path: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[dict]:
    lines = load_persona_chat_lines(split=split, local_json_path=local_json_path, max_examples=None)
    conversations = []
    example_id = 1

    for text in lines:
        parsed = parse_persona_text(text)
        if parsed is None:
            continue

        input_text, output_text = parsed
        conversations.append(
            {
                "id": example_id,
                "input": input_text,
                "output": output_text,
            }
        )
        example_id += 1
        if max_examples and len(conversations) >= max_examples:
            break

    return conversations


def save_persona_chat_lines(
    split: str = "train",
    output_path: str = "data/persona_chat_lines.json",
    max_examples: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = load_persona_chat_lines(split=split, max_examples=max_examples)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)

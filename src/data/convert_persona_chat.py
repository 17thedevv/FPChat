import argparse
import json
import os
import re

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets package is required. Install it with: pip install datasets"
    )


def parse_persona_text(text: str):
    """Parse a single persona-chat line into (input, output) pairs."""
    text = text.strip()
    if not text:
        return None

    m = re.match(r"^\d+\s+(.*)$", text)
    if not m:
        return None

    body = m.group(1).strip()
    parts = body.split("\t")
    if len(parts) < 2:
        return None

    input_text = parts[0].strip()
    output_text = parts[1].strip()
    if not input_text or not output_text:
        return None

    # Skip persona definition lines that are not actual conversation turns
    if "your persona:" in input_text or "partner's persona:" in input_text:
        return None

    return input_text, output_text


def build_intents_from_split(split: str, max_examples: int):
    dataset = load_dataset("awsaf49/persona-chat", split=split)
    conversations = []
    seen_inputs = set()
    example_id = 1

    for item in dataset:
        if len(conversations) >= max_examples:
            break

        parsed = parse_persona_text(item["text"])
        if parsed is None:
            continue

        input_text, output_text = parsed
        if input_text in seen_inputs:
            continue

        seen_inputs.add(input_text)
        conversations.append(
            {
                "id": example_id,
                "input": input_text,
                "output": output_text,
                "category": output_text,
            }
        )
        example_id += 1

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert persona-chat into an intents.json dataset for FPChat."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Which persona-chat split to convert.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2000,
        help="Maximum number of conversation examples to include.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "intents.json"),
        help="Output file path for the converted intents dataset.",
    )
    parser.add_argument(
        "--save-raw",
        default=None,
        help="Optional path to save raw persona-chat lines as JSON.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading persona-chat split={args.split}...")
    conversations = build_intents_from_split(args.split, args.max_examples)
    if not conversations:
        raise RuntimeError(
            "Không tìm thấy cặp input/output nào trong persona-chat."
        )

    data = {"conversations": conversations}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversations)} examples to {args.output}")

    if args.save_raw:
        raw_path = args.save_raw
        raw_data = []
        dataset = load_dataset("awsaf49/persona-chat", split=args.split)
        for item in dataset:
            raw_data.append(item["text"])
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        print(f"Saved raw persona-chat lines to {raw_path}")


if __name__ == "__main__":
    main()

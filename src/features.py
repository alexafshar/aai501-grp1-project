"""Create measurable RAGTruth answer features for Group 01 task T07.

Run from the project root:
    python src/features.py

Reads data/processed/responses_clean.csv and writes
    data/processed/features.csv

No hallucination labels are used to *calculate* the features.
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd


WORD_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
NUMBER_PATTERN = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)*(?!\w)")
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*\b")


def text_from_source(source_info):
    """Turn string or structured RAGTruth source information into searchable text."""
    if source_info is None:
        return ""
    if isinstance(source_info, (dict, list)):
        values = source_info.values() if isinstance(source_info, dict) else source_info
        return " ".join(text_from_source(value) for value in values)
    if isinstance(source_info, str):
        stripped = source_info.strip()
        if stripped.startswith(("{", "[")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    if isinstance(parsed, (dict, list)):
                        return text_from_source(parsed)
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
        return source_info
    if pd.isna(source_info):
        return ""
    return str(source_info)


def words(text):
    """Return lowercase word tokens from an answer or source."""
    return WORD_PATTERN.findall(str(text).lower())


def word_overlap(answer, source):
    """Return the fraction of answer word tokens also present in the source."""
    answer_words = words(answer)
    if not answer_words:
        return 0.0
    source_words = set(words(source))
    return sum(word in source_words for word in answer_words) / len(answer_words)


def missing_numbers(answer, source):
    """Count distinct numeric strings in the answer not found in the source."""
    return len(set(NUMBER_PATTERN.findall(str(answer))) -
               set(NUMBER_PATTERN.findall(str(source))))


def missing_names(answer, source):
    """Count distinct capitalized words in the answer absent from the source.

    This is a rough heuristic, not a named-entity recognizer; it can flag a
    sentence-opening word or miss names with unusual capitalization.
    """
    answer_names = set(NAME_PATTERN.findall(str(answer)))
    source_lower = {name.lower() for name in NAME_PATTERN.findall(str(source))}
    return sum(name.lower() not in source_lower for name in answer_names)


def answer_length(answer):
    """Count the number of word tokens in the generated answer."""
    return len(words(answer))


def length_ratio(answer, source):
    """Return answer token length divided by source token length, or zero."""
    source_length = len(words(source))
    return answer_length(answer) / source_length if source_length else 0.0


def build_features(df):
    """Compute numeric features and one-hot encode task type and model."""
    required = {"response", "source_info", "task_type", "model"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cleaned CSV lacks required columns: {missing}")

    # Preserve identifiers and labels for downstream evaluation, not as predictors.
    metadata = [c for c in ("id", "source_id", "split", "hallucinated")
                if c in df.columns]
    features = df[metadata].copy().reset_index(drop=True)
    records = []
    for answer, source_info in zip(df["response"], df["source_info"]):
        answer = "" if pd.isna(answer) else str(answer)
        source = text_from_source(source_info)
        records.append({
            "word_overlap": word_overlap(answer, source),
            "missing_numbers": missing_numbers(answer, source),
            "missing_names": missing_names(answer, source),
            "answer_length": answer_length(answer),
            "length_ratio": length_ratio(answer, source),
        })
    features = pd.concat([features, pd.DataFrame(records)], axis=1)
    encoded = pd.get_dummies(
        df[["task_type", "model"]].fillna("Unknown").reset_index(drop=True),
        columns=["task_type", "model"],
        dtype=int,
    )
    return pd.concat([features, encoded], axis=1)


def main():
    """Read the T04 output, build T07 features, and save the results."""
    input_file = Path("data/processed/responses_clean.csv")
    output_file = Path("data/processed/features.csv")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing {input_file}; put your T04 CSV there.")
    df = pd.read_csv(input_file, low_memory=False)
    features = build_features(df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)
    print(f"Input rows: {len(df):,}")
    print(f"Feature rows: {len(features):,}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Missing values: {int(features.isna().sum().sum())}")
    print(f"Saved to: {output_file}")
    print(features.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

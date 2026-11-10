# AAI 501 Group 01 Project

## RAGTruth dataset download

This project uses the RAGTruth hallucination benchmark from ParticleMedia.

The dataset files are included in this repository under `data/raw/` so team members can pull the project and work with the same data immediately.

Reference: https://github.com/ParticleMedia/RAGTruth/blob/main/README.md

1. If you want the project files from GitHub, pull the repository:

```bash
git pull origin main
```

2. Load the files with pandas and verify the expected row counts:

```bash
python
```

```python
import pandas as pd

response_df = pd.read_json("data/raw/response.jsonl", lines=True)
source_df = pd.read_json("data/raw/source_info.jsonl", lines=True)

print(len(response_df))
print(len(source_df))
```

Expected output:

```text
17790
2965
```

Side note: if a teammate wants to refresh the dataset from the upstream source instead of using the repository copy, they can follow the original RAGTruth instructions and copy the files into `data/raw/` manually.

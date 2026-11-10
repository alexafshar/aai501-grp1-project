# AAI 501 Group 01 Project

## RAGTruth dataset download

**This project uses the RAGTruth hallucination benchmark from ParticleMedia.** [RAGTruth README](https://github.com/ParticleMedia/RAGTruth/blob/main/README.md)

The dataset files are included in this repository under `data/raw/` so team members can pull the project and work with the same data immediately.

1. If you are starting from scratch, clone the project repository to your computer:

```bash
git clone https://github.com/alexafshar/aai501-grp1-project.git
```

Then move into the project folder:

```bash
cd aai501-grp1-project
```

2. To update your local repository with the latest project files from GitHub, pull the repository:

```bash
git pull origin main
```

3. Load the files with pandas and verify the expected row counts. You can do this either in a Python session or by opening the notebook at `notebooks/01_load_ragtruth_data.ipynb`.

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

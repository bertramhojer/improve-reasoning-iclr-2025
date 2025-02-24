# Improving "Reasoning" in Large Language Models via Representation Engineering

This repository comprises the code that accompanies the paper [Improving "Reasoning" in Large Language Models via Representation Engineering](https://openreview.net/pdf?id=IssPhpUsKt) presented at ICLR 2025. We apply control vectors to modulate the reasoning performance of various LLMs.

## Instructions

We developed `repana` to extract hidden states and derive control vectors. The `repana` framework is based on [repeng](https://github.com/vgel/repeng) (MIT License), a very nice library for training control vectors!

### Installation

We manage our environment using `uv`. 

To install `uv` run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To set up the project run:
```bash
git clone <repository-url>
cd <project-name>
```

Create a virtual environment and activate it:
```bash
uv venv
source .venv/bin/activate  # On Unix-like systems
```
Install dependencies:
```bash
# Install from requirements.txt
uv pip install -r requirements.txt
```

### Replication

To replicate our experiments follow these instructions:

## Citation

Please cite this work as:

```
@inproceedings{hojerImprovingReasoningPerformanceLargeLanguage2025,
  title = {Improving {{Reasoning Performance}} in {{Large Language Models}} via {{Representation Engineering}}},
  author = {Højer, Bertram and Jarvis, Oliver and Heinrich, Stefan},
  date = {2025-01-02},
  eventtitle = {International {{Conference}} on {{Learning Representations}}},
  langid = {english}
}
```

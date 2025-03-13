# Improving "Reasoning" in Large Language Models via Representation Engineering

This repository comprises the code that accompanies the paper [Improving "Reasoning" in Large Language Models via Representation Engineering](https://openreview.net/pdf?id=IssPhpUsKt) presented at ICLR 2025. We apply control vectors to modulate the reasoning performance of various LLMs.

## Instructions

We developed `repana` to extract hidden states and derive control vectors. The `repana` framework is based on [repeng](https://github.com/vgel/repeng) (MIT License), a very nice library for training control vectors!

### Installation

We manage our environment using `uv`. Assuming you've cloned and navigated to the repository (and don't have `uv` installed), follow these steps to set up your environment (unix systems).

To install `uv` run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

Our experiments were run using the HPC at the IT University of Copenhagen. It was unfeasible to run the experiments with e.g. 7B parameter models locally. If you have a GPU avaiable the `repana` framework should automatically push your model to GPU, enabling you to run experiments with larger models.

### Replication

We manage hyperparameters using Hydra. To replicate our experiments follow these instructions:

Deriving control-vectors based on the hyperparameters for experiment 1.
```bash
uv run train experiment=experiment1
```

Evaluating control-vectors based on the hyperparameters for experiment 1.
```bash
uv run evaluate experiment=experiment1
```

Experiments were run on the HPC provided by the IT University of Copenhagen. Experiments were run using singularity containers on the HPC.
To run experiments using the same process on a different cluster you can build a docker image using the provided dockerfile:
```bash
docker buildx build --platform=linux/amd64 --load -t experiment-image:latest .
```

Save the docker image as a tar ball.
```bash
docker save experiment-image:latest -o experiment.tar
```

Either build the singularity container locally and push it an available HPC or push the .tar there and build the singularity container on the cluster.

```bash
singularity build experiment.sif docker-archive://experiment.tar
```

## Citation

Please cite this work as:

```
@inproceedings{hojerImprovingReasoningPerformanceLargeLanguage2025,
  title = {Improving Reasoning Performance in Large Language Models via Representation Engineering},
  author = {Højer, Bertram and Jarvis, Oliver and Heinrich, Stefan},
  date = {2025-01-02},
  eventtitle = {The Thirteenth International {{Conference}} on {{Learning Representations}}},
  langid = {english}
}
```

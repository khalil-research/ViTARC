# setup.py
from setuptools import setup, find_packages

setup(
    name="ViTARC",
    version="0.1.0",
    packages=find_packages(),  # Automatically discovers directories with __init__.py
    python_requires=">=3.10",
    install_requires=[
        # Pin numpy to a <2.0 version but >=1.20
        "numpy>=1.20,<2.0",        
        "torch>=2.3",
        "pytorch-lightning>=2.1",
        "transformers==4.44.2",
        "datasets==2.20.0",
        "scikit_learn>=1.0",
        "sentencepiece>=0.1.0",
        "tokenizers>=0.10",
        "sacrebleu>=2.3",
        "opencv-python>=4.10.0",
        "huggingface-hub>=0.16",
        "filelock>=3.0",
        "fsspec>=2023.1",
        "wandb>=0.18.0",
        "prettytable>=3.11",
        "nltk>=3.9",
        "seaborn>=0.13.2",
        "tqdm>=4.66.5",
    ],
    extras_require={
        "dev": [
            "pytest",       # For unit tests
            "pytest-cov",   # For coverage
        ],
    },
    author="Wenhao Li",
    description="ViTARC: A custom ARC dataset generation and T5-based 2D-ViT model project.",
    url="https://github.com/khalil-research/ViTARC",
    license="MIT",
)

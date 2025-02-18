# setup.py
from setuptools import setup, find_packages

setup(
    name="ViTARC",
    version="0.1.0",
    packages=find_packages(),  # Automatically finds any folder with __init__.py
    python_requires=">=3.10",
    install_requires=[
        # Below are some commonly used packages from your requirements.txt.
        # Adjust or pin exact versions if desired.
        "aiohttp>=3.11.12",
        "datasets>=2.20.0",
        "numpy>=2.2.3",
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "opencv-python>=4.11.0.86",
        "prettytable>=3.14.0",
        "pytorch-lightning>=2.5.0",
        "scikit-learn>=1.6.1",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.21.0",
        "tqdm>=4.67.1",
        "pyarrow>=19.0.0",
    ],
    # You can also add extra metadata if you like:
    author="Wenhao Li",
    description="ViTARC: A custom ARC dataset generation and T5-based 2D-ViT model project.",
    url="https://github.com/khalil-research/ViTARC",
    license="MIT",  # or whichever you use
)

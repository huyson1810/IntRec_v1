from setuptools import setup, find_packages

setup(
    name="recbole-sae",
    version="0.1.0",
    description="Sparse Autoencoder interpretation layer for RecBole (RecSAE, Wang et al. 2026)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "recbole>=1.2.0",
        "torch>=1.13.0",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "hf":     ["transformers>=4.40", "accelerate"],
        "openai": ["openai>=1.0"],
    },
)

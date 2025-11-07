from setuptools import setup, find_packages

setup(
    name="sae_captioning",
    version="1.0.0",
    description="SAE-based Mechanistic Interpretability for Cross-Lingual Image Captioning Bias",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "arabic": [
            "camel-tools>=1.5.0",
            "pyarabic>=0.6.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "sae-extract=scripts.02_extract_activations:main",
            "sae-train=scripts.03_train_sae:main",
            "sae-analyze=scripts.04_analyze_features:main",
        ],
    },
)

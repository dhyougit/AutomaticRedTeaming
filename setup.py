"""
Setup script for LLM Red Team Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-redteam",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated red teaming framework for testing LLM safety and security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-red-team",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.9.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "jinja2>=3.1.0",
        "markdown>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
        "openai": ["openai>=1.6.0"],
        "anthropic": ["anthropic>=0.8.0"],
        "local": ["transformers>=4.36.0", "torch>=2.1.0"],
    },
    entry_points={
        "console_scripts": [
            "llm-redteam=auto_redteam:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "llm",
        "security",
        "red-team",
        "ai-safety",
        "adversarial-testing",
        "machine-learning",
        "nlp"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-red-team/issues",
        "Source": "https://github.com/yourusername/llm-red-team",
        "Documentation": "https://github.com/yourusername/llm-red-team/blob/main/README.md",
    },
)

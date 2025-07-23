from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="measure",
    version="0.1.0",
    author="Simon Meoni",
    author_email="simonmeoni@aol.com",
    description="A Python library for measurements and calculations",
    url="https://github.com/simonmeoni/measure",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "accelerate==1.7.0",
        "bert-score==0.3.13",
        "datasets==3.6.0",
        "evaluate==0.4.3",
        "mauve-text==0.4.0",
        "openai==1.66.3",
        "pandas>=1.3.0",
        "protobuf==6.31.1",
        "pycanon==1.0.5",
        "python-dotenv==1.0.0",
        "rouge_score==0.1.2",
        "sacrebleu==2.5.1",
        "sentence-transformers==4.1.0",
        "sentencepiece==0.2.0",
        "tiktoken==0.9.0",
        "tokenizers==0.21.1",
        "torch==2.6.0",
        "transformers==4.50.0",
    ],
)

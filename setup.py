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
        "pandas>=1.3.0",
        "evaluate==0.4.3",
        "sacrebleu==2.5.1",
        "rouge_score==0.1.2",
        "bert-score==0.3.13",
        "mauve-text==0.4.0",
        "sentence-transformers==4.1.0",
        "accelerate==1.7.0",
    ],
)

from pathlib import Path
from setuptools import find_packages, setup

here = Path(__file__).parent
setup(
    name="aimed-media",
    version="0.1.0",
    author="Felipe Meneguitti Dias, Estela Ribeiro, Quenaz Bezerra Soares",
    description="Training utilities for AIMED PhysioNet Challenge 2024",
    long_description=(here / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
)

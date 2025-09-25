from pathlib import Path
from setuptools import setup, find_packages

requirements = Path(__file__).with_name("requirements.txt").read_text().splitlines()

setup(
    name="yaiba_bi",
    version="0.0.0",
    description="Tools for analyzing and visualizing YAIBA VRChat logs",
    author="YAIBA Democratization Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
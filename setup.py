from setuptools import setup, find_packages

setup(
    name="bayesian_equalizer",
    version="0.1.0",
    description="Bayesian Nonparametric Equalizer with SMC inference and DP-GP priors",
    author="Oluwaseyi Giwa",
    author_email="oluwaseyi@aims.ac.za",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "omegaconf",
        "wandb"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
)

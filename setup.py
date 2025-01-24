from setuptools import find_packages, setup

setup(
    name="facto",
    version="0.1.0",
    author="Manuel Candales",
    description="Framework for Algorithmic Correctness Testing of Operators",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/FACTO",
    packages=find_packages(include=["facto", "facto.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["torch"],
)

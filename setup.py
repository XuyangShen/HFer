import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="hfer",
    version="0.0.4",
    author="Ryan",
    author_email="xuyangshen1122@gmail.com",
    description="one-line hf model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XuyangShen/HFer",
)

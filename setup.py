from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as file:
    long_description = file.read()

__version__ = "0.0.0"

REPO_NAME = "kidney-disease-classification"
AUTHOR_USER_NAME = "ShamsudeenLawal"
SRC_REPO = "kidney_disease_classifier"
AUTHOR_EMAIL = "sendtolawal@gmail.com"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for kidney disease classification",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
import setuptools

with open ("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__  = "0.0.0"

REPO_NAME = "Kidney-Disease-Classification-DeepLearning-Project"
AUTHOR_USER_NAME = "Sourav Halder"
SRC_REPO = "Kidney-Disease-Classifier"
AUTHOR_EMAIL = "halder.sourav1996@gmai.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_enail = AUTHOR_EMAIL,
    description = "A deep CNN based model for detecting Kidney Disease",
    long_description = long_description,
    url = f"https://github.com/SouravHalder1996/{REPO_NAME}"
)
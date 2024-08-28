import shutil
import os

shutil.rmtree("_examples")

try:
    os.remove("README.md")
except:
    pass
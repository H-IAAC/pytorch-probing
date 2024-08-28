import shutil
import os

shutil.rmtree("_examples")
shutil.rmtree("auto_doc")

try:
    os.remove("README.md")
except:
    pass
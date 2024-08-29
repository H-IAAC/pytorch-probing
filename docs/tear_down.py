import shutil
import os


for path in ["_examples", "auto_doc"]:
    try:
        shutil.rmtree(path)
    except:
        pass

try:
    os.remove("README.md")
except:
    pass
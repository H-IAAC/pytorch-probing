import subprocess
import sys
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join("..", "setup.cfg"))
packages = config["options.extras_require"]["doc_generation"]

subprocess.check_call([sys.executable, "-m", "pip", "install"]+packages.split(";"))
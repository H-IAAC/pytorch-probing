import os
import glob
import subprocess
import unittest
import pathlib
import sys
from typing import List




class TestTypeCheck(unittest.TestCase):

    def test_run_mypy_module(self):
        """Run mypy on all module sources"""
        mypy_call: List[str] = self.base_call + [self.pkg_path]
        subprocess.check_call(mypy_call)

    #def test_run_mypy_tests(self):
    #    """Run mypy on all tests in module under the tests directory"""
    #    mypy_call: List[str] = self.base_call + [self.tests_path]
    #    subprocess.check_call(mypy_call)

    def __init__(self, *args, **kwargs) -> None:
        super(TestTypeCheck, self).__init__(*args, **kwargs)
        
        self.tests_path = pathlib.Path(__file__).parent.resolve()


        self.pkg_path = os.path.join(self.tests_path, "../src/pytorch_probing")

        self.mypy_opts: List[str] = ['--ignore-missing-imports']

        self.base_call : List[str] = [sys.executable, "-m", "mypy"] + self.mypy_opts
        
        
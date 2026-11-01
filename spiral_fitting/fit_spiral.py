import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SUBDIR = Path(__file__).resolve().parent / 'villa_8aa507c'

sys.path.insert(0, str(SUBDIR))
os.environ['PYTHONPATH'] = str(SUBDIR) + os.pathsep + os.environ.get('PYTHONPATH', '')

runpy.run_module('fit_spiral', run_name='__main__', alter_sys=True)
import os
import glob

from .modules import *

# import all python files as module without importing one by one
# https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
# files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
# __all__ = [os.path.basename(f)[:-3] for f in files if os.path.isfile(f) and not f.endswith('__init__.py')]
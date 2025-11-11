import os
import pkgutil
import importlib

for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

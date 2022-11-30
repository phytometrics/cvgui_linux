import importlib, inspect

# file path should be relative to main.py
module_files = ["image_analysis_modules",
                ""]

for name, cls in inspect.getmembers(importlib.import_module("image_analysis_modules"), inspect.isclass):
    print(cls)
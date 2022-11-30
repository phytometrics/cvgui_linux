import importlib, inspect

for name, cls in inspect.getmembers(importlib.import_module("image_analysis_modules"), inspect.isclass):

    # change to a more smarter way

        print(cls)

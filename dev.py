import image_analysis_modules
import importlib
import inspect

def get_ia_module_dict():
    ia_module_dict = {}
    for name, cls in inspect.getmembers(importlib.import_module("image_analysis_modules"), inspect.isclass):
        # import modules that are directly defined in __init__ under image_analysis_modules
        if cls.__module__.split(".")[-1] == "image_analysis_modules":
            ia_module_dict[cls.__name__] = {}
            ia_module_dict[cls.__name__]["func"] = cls
            ia_module_dict[cls.__name__]["description"] = cls.description
            ia_module_dict[cls.__name__]["vis"] = cls.vis
            #print(cls.__base__)
    return ia_module_dict

def list_ia_module_dict():
    print("list")
    for name, cls in inspect.getmembers(importlib.import_module("image_analysis_modules"), inspect.isclass):
        if cls.__base__.__module__.split(".")[-1] == "BaseModule":
            pass

list_ia_module_dict()
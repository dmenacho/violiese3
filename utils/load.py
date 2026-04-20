import importlib
from typing import Callable, List, Optional

def get_callable(module_name: str, callable_name: str) -> Callable:
    """Returns callable function or method from the module listed.

    callable_name can either be: "function_name" or "class_name.method_name"
    """
    if "." in callable_name:
        assert callable_name.count(".") == 1, "Only one dot is allowed in callable_name"
        class_name, function_name = callable_name.split(".")
    else:
        class_name, function_name = None, callable_name

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Module {module_name} not found. Error: {e}")

    obj = module
    if class_name:
        try:
            obj = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in module {module_name}. Error: {e}")

    try:
        candidate = getattr(obj, function_name)
    except AttributeError:
        raise ImportError(f"Function or method '{function_name}' not found in '{module_name}'")

    if not callable(candidate):
        raise TypeError(f"'{function_name}' in '{module_name}' is not callable")

    return candidate

from pathlib import Path
import importlib

def package_available(package_name: str, fn: callable = None, file = None) -> bool:
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        if file:
            msg = f"Need to install {package_name} in order to import from file '{Path(file).stem}'."
        elif fn:
            msg = f"Need to install {package_name} in order to use function '{fn.__name__}'."
        else:
            msg = f"Need to install {package_name}."
        raise ModuleNotFoundError(msg)
    else:
        pass



import functools
from importlib.util import find_spec


class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise KeyError(key)
        return builder(**kwargs)


def package_available(package_name: str) -> bool:
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def check_package(package_name: str, file: str = None) -> None:
    """_summary_

    If `file` is not provided, it assumes you're using a decorator and
    the corresponding error message will reflect that.

    If `file` if provided, it assumes you're using this as a standalone
    function in a file and the corresponding error message will reflect
    that.

    Args:
        package_name (str): _description_
        file (str, optional): _description_. Defaults to None.

    Raises:
        ModuleNotFoundError: _description_
        ModuleNotFoundError: _description_
    """
    if not package_available(package_name):
        if file:
            raise ModuleNotFoundError(
                f"Need to install '{package_name}' to use functions from '{file}'"
            )
        else:
            raise ModuleNotFoundError(
                f"Need to install '{package_name}' to use this function."
            )


def check_package_decorator(package_name: str, filename: str = None):
    def _check_package_decorator(func):
        @functools.wraps(func)
        def check_package_wrapper(*args, **kwargs):
            check_package(package_name, filename)
            return func(*args, **kwargs)

        return check_package_wrapper

    return _check_package_decorator

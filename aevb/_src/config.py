from typing import List

import json
from pathlib import Path

def _flatten(mapping, sep):
    result = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            for k, v in _flatten(value, sep).items():
                combined = f"{key}{sep}{k}"
                result[combined] = v
        else:
            result[key] = value
    return result


def _nest(mapping, sep):
    result = {}
    for key, value in mapping.items():
        parts = key.split(sep)
        node = result
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def _ensure_values(mapping):
    result = json.loads(json.dumps(mapping))
    for key, value in result.items():
        if isinstance(value, list):
            value = tuple(value)
        if isinstance(value, tuple):
            if len(value) == 0:
                message = f"'{key} : {value}'. Empty lists are disallowed because their type is unclear."
                raise TypeError(message)
            if not isinstance(value[0], (str, float, int, bool)):
                message = f"'{key} : {value}'. Lists can only contain strings, floats, ints, bools"
                message += f" but not {type(value[0])}"
                raise TypeError(message)
            if not all(isinstance(x, type(value[0])) for x in value[1:]):
                message = f"'{key} : {value}'. Elements of a list must all be of the same type."
                raise TypeError(message)
        result[key] = value
    return result



def _format_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in value) + "]"
    return str(value)


def _format_type(value):
    if isinstance(value, (list, tuple)):
        assert len(value) > 0, value
        return _format_type(value[0]) + "s"
    return str(type(value).__name__)



class Config(dict):

    SEP = "."

    def __init__(self, *args, **kwargs):
        mapping = dict(*args, **kwargs)
        mapping = _flatten(mapping, self.SEP)
        mapping = _ensure_values(mapping)
        self._flat = mapping
        self._nested = _nest(mapping, self.SEP)
        super().__init__(self._nested)



    @property
    def flat(self):
        return self._flat
    
    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattr__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        result = self._nested
        for part in name.split(self.SEP):
            try:
                result = result[part]
            except TypeError:
                raise KeyError
        if isinstance(result, dict):
            result = type(self)(result)
        return result

    def __setattr__(self, key, value):
        if key.startswith("_"):
            return super().__setattr__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __setitem__(self, key, value):
        if key.startswith("_"):
            return super().__setitem__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)
    
    def __reduce__(self):
        return (type(self), (dict(self),))


    def prettyprint(self):
        lines = ["\nConfig:"]
        keys, vals, typs = [], [], []
        for key, val in self.flat.items():
            keys.append(key + ":")
            vals.append(_format_value(val))
            typs.append(_format_type(val))
        max_key = max(len(k) for k in keys) if keys else 0
        max_val = max(len(v) for v in vals) if vals else 0
        for key, val, typ in zip(keys, vals, typs):
            key = key.ljust(max_key)
            val = val.ljust(max_val)
            lines.append(f"{key}  {val}  ({typ})")
        print("\n".join(lines))


    def update(self, *args, **kwargs):
        result = self._flat.copy()
        inputs = _flatten(dict(*args, **kwargs), self.SEP)
        for key, new in inputs.items():
            keys = [key]
            if not keys:
                raise KeyError(f"Unknown key or pattern {key}.")
            for key in keys:
                old = result[key]
                try:
                    if isinstance(old, int) and isinstance(new, float):
                        if float(int(new)) != new:
                            message = f"Cannot convert fractional float {new} to int."
                            raise ValueError(message)
                    result[key] = type(old)(new)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Cannot convert '{new}' to type '{type(old).__name__}' "
                        + f"for key '{key}' with previous value '{old}'."
                    )
        return type(self)(result)

    # TODO: can probably bring the override out and make it more robust
    @classmethod
    def from_yaml(cls, fp, override: str | List[str] = None):
        import ruamel.yaml as yaml
        yaml = yaml.YAML(typ="safe")
        data = yaml.load(Path(fp))
        to_return = data["defaults"]
        if override:
            if isinstance(override, str):
                override = (override, )
            for override_item in override:
                override_with = data[override_item]
                for k in to_return.keys():
                    if k in override_with.keys():
                        to_return[k] = override_with[k]
        else:
            to_return = data["defaults"]
        return cls(to_return)







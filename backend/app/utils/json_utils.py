"""
Fast JSON utilities using orjson.

orjson is 6x faster than standard json library.
This module provides compatibility wrappers since orjson.dumps() returns bytes.
"""

import orjson
from typing import Any, IO, Union
from pathlib import Path


def dumps(obj: Any, indent: bool = False, **kwargs) -> str:
    """
    Serialize object to JSON string.

    Args:
        obj: Object to serialize
        indent: If True, format with indentation (2 spaces)
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        JSON string
    """
    options = orjson.OPT_NON_STR_KEYS
    if indent:
        options |= orjson.OPT_INDENT_2

    return orjson.dumps(obj, option=options).decode('utf-8')


def dump(obj: Any, fp: IO, indent: int = None, **kwargs) -> None:
    """
    Serialize object to JSON file.

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        indent: If provided, format with indentation
        **kwargs: Additional arguments (ignored for compatibility)
    """
    options = orjson.OPT_NON_STR_KEYS
    if indent:
        options |= orjson.OPT_INDENT_2

    fp.write(orjson.dumps(obj, option=options).decode('utf-8'))


def loads(s: Union[str, bytes]) -> Any:
    """
    Deserialize JSON string to object.

    Args:
        s: JSON string or bytes

    Returns:
        Deserialized object
    """
    return orjson.loads(s)


def load(fp: IO) -> Any:
    """
    Deserialize JSON file to object.

    Args:
        fp: File-like object to read from

    Returns:
        Deserialized object
    """
    return orjson.loads(fp.read())


def load_file(path: Union[Path, str]) -> Any:
    """
    Load JSON from file path.

    Args:
        path: Path to JSON file

    Returns:
        Deserialized object
    """
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def save_file(obj: Any, path: Union[Path, str], indent: bool = True) -> None:
    """
    Save object to JSON file.

    Args:
        obj: Object to serialize
        path: Path to save to
        indent: If True, format with indentation
    """
    options = orjson.OPT_NON_STR_KEYS
    if indent:
        options |= orjson.OPT_INDENT_2

    with open(path, 'wb') as f:
        f.write(orjson.dumps(obj, option=options))

"""Compatibility layer for Pydantic v1 and v2.

We don't want to pin any specific version of Pydantic. With this, we can
import things from `zeus.utils.pydantic_v1` and always use the V1 API
regardless of the installed version of Pydantic.

Inspired by Deepspeed:
https://github.com/microsoft/DeepSpeed/blob/5d754606/deepspeed/pydantic_v1.py
"""

# pyright: reportWildcardImportFromLibrary=false

try:
    from pydantic.v1 import *
except ImportError:
    from pydantic import *

"""Test the DeprecatedAliasABCMeta metaclass and @deprecated_alias decorator."""

from __future__ import annotations

import abc
import pytest
from dataclasses import dataclass

from zeus.device.common import DeprecatedAliasABCMeta, deprecated_alias


class MockDeviceBase(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    def __init__(self) -> None:
        self.value = 0

    @deprecated_alias("getName")
    @abc.abstractmethod
    def get_name(self) -> str: ...

    @deprecated_alias("getValue")
    @abc.abstractmethod
    def get_value(self) -> int: ...

    @deprecated_alias("setValue")
    @abc.abstractmethod
    def set_value(self, value: int) -> None: ...


class MockDevice(MockDeviceBase):
    def get_name(self) -> str:
        return "Mock Device"

    def get_value(self) -> int:
        return self.value

    def set_value(self, value: int) -> None:
        self.value = value


@dataclass
class MockDataclassBase(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    @deprecated_alias("zeroAllFields")
    @abc.abstractmethod
    def zero_all_fields(self) -> None: ...


@dataclass
class MockDataclass(MockDataclassBase):
    value: int = 0

    def zero_all_fields(self) -> None:
        self.value = 0


def test_dataclass_subclass_is_instantiable():
    obj = MockDataclass(value=42)
    obj.zero_all_fields()
    assert obj.value == 0


def test_method_calls():
    device = MockDevice()

    # Test new snake_case method
    assert device.get_name() == "Mock Device"

    with pytest.deprecated_call():
        assert device.getName() == "Mock Device"  # type: ignore

    device.set_value(42)
    assert device.get_value() == 42
    with pytest.deprecated_call():
        device.setValue(100)  # type: ignore
    assert device.get_value() == 100
    with pytest.deprecated_call():
        assert device.getValue() == 100  # type: ignore

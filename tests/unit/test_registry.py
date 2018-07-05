# pylint: skip-file
import pytest
from galini.registry import Registry

class _MockClass:
    pass


class _MockEntryPoint:
    def __init__(self, name):
        self.name = name

    def load(self):
        return _MockClass


class _MockRegistry(Registry):
    def __init__(self, names):
        self.names = names
        super().__init__()

    def group_name(self):
        return 'test_registry'

    def iter_entry_points(self):
        for name in self.names:
            yield _MockEntryPoint(name)


def test_registry_returns_a_class():
    reg = _MockRegistry(['one', 'two', 'three'])
    assert len(reg.keys()) == 3
    assert isinstance(reg.get('three'), type)


def test_registry_raises_exception_on_duplicate_entry_points():
    with pytest.raises(SystemExit) as exc:
        reg = _MockRegistry(['one', 'two', 'three', 'one'])
    assert exc.type == SystemExit

"""Registry module."""
import abc
import logging
import sys
import pkg_resources


class Registry(metaclass=abc.ABCMeta):
    """Registry for pkg_resources entry points."""
    def __init__(self):
        self.group = self.group_name()
        self._registered = {}
        for entry_point in pkg_resources.iter_entry_points(self.group_name()):
            if entry_point.name in self._registered:
                logging.error(
                    'Duplicate registered item %s found in %s registry.',
                    entry_point.name, self.group,
                )
                sys.exit(1)
            obj_cls = entry_point.load()
            self._registered[entry_point.name] = obj_cls

    def get(self, name, default=None):
        """Return entry point associated with name."""
        return self._registered.get(name, default)

    def keys(self):
        """Return the registered objects names."""
        return self._registered.keys()

    @abc.abstractmethod
    def group_name(self):
        """Return this registry group name."""
        pass

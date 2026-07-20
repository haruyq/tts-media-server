import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.exceptions import PluginNotFound
from utils.plugins import PluginManager

class PluginManagerTest(unittest.TestCase):
    def test_loads_plugins(self):
        plugin = object()
        entry = SimpleNamespace(name="test", load=lambda: plugin)

        with patch("utils.plugins.entry_points", return_value=[entry]):
            manager = PluginManager()

        self.assertEqual(manager.names, ["test"])
        self.assertIs(manager.get("test"), plugin)

        with self.assertRaises(PluginNotFound):
            manager.get("missing")

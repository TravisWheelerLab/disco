"""
Module includes methods useful to loading all plugins placed in a folder, or module...
"""
from __future__ import annotations

import pkgutil
import sys
import warnings
from types import ModuleType
from typing import TypeVar

warnings.simplefilter("always", ImportWarning)

# Generic type for method below
T = TypeVar("T")


def load_plugin_classes(
    plugin_dir: ModuleType,
    plugin_metaclass: type[T],
    do_reload: bool = True,
    display_error: bool = True,
) -> set[type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder and submodules that extend the provided metaclass
    type.

    :param plugin_dir: A module object representing the path containing plugins... Can get a module object
                       using import...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.
    :param do_reload: Boolean, Determines if plugins should be reloaded if they already exist. Defaults to True.
    :param display_error: Boolean, determines if import errors are sent using python's warning system when they occur.
                          Defaults to True. Note these warnings won't be visible unless you set up a filter for them,
                          such as below:

                          import warnings
                          warnings.simplefilter("always", ImportWarning)

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
    # Get absolute and relative package paths for this module...
    path = list(iter(plugin_dir.__path__))[0]
    rel_path = plugin_dir.__name__

    plugins: set[type[T]] = set()

    # Iterate all modules in specified directory using pkgutil, importing them if they are not in sys.modules
    for importer, mod_name, ispkg in pkgutil.iter_modules([path], rel_path + "."):
        # If the module name is not in system modules or the reload flag is set to true, perform a full load of the
        # modules...
        if (mod_name not in sys.modules) or do_reload:
            try:
                sub_module = importer.find_module(mod_name).load_module(mod_name)
                sys.modules[mod_name] = sub_module
            except Exception as e:
                print(e)
                if display_error:
                    import traceback

                    warnings.warn(
                        f"Can't load '{mod_name}'. Due to issue below: \n {traceback.format_exc()}",
                        ImportWarning,
                    )
                continue
        else:
            sub_module = sys.modules[mod_name]

        # Now we check if the module is a package, and if so, recursively call this method
        if ispkg:
            plugins = plugins | load_plugin_classes(
                sub_module,
                plugin_metaclass,
                do_reload,
            )
        else:
            # Otherwise we begin looking for plugin classes
            for item in dir(sub_module):
                field = getattr(sub_module, item)

                # Checking if the field is a class, and if the field is a direct child of the plugin class
                try:
                    if (
                        isinstance(field, type)
                        and issubclass(field, plugin_metaclass)
                        and (field != plugin_metaclass)
                    ):
                        # It is a plugin, add it to the list...
                        plugins.add(field)
                except Exception:
                    # For some reason passing some instances of 'type' to issubclass still throws an error, so we skip over those...
                    continue

    return plugins

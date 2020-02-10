"""Dummy module for pipeline runner example.

This module's only purpose is to demonstrate the functionality and usage
of the pipeline runner module. It is only used in the example config file
'example_config.yaml'.
"""

# built-in modules:
from pprint import pprint


def dummy_fun_1():
    """Dummy function 1.

    Args:
        None.

    """
    return None


def dummy_fun_2():
    """Dummy function 2.

    Args:
        None.

    """
    return None


def dummy_data_handler(data):
    """Dummy data handler.

    Args:
        data (any type): Some form of data. The function will only print and
            return it.

    Returns:
        data (any type): Input data.

    """
    print("dummy_data_handler called, with data:")
    pprint(data)
    print('\n')
    return data

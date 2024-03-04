from collections.abc import MutableMapping


def flatten(dictionary: MutableMapping, parent_key: str = "", separator: str = "/"):
    """Flatten a nested dictionary.
    Args:
        dictionary (dict): The dictionary to flatten.
        parent_key (str): The parent key of the dictionary.
        separator (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

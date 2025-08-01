import numpy as np


def merge_dict(dict1, dict2):
    """
    Merge two dictionaries, summing the values of any
    keys they share.

    Parameters
    ----------
    dict1: Dict[Variable, np.ndarray | np.matrix]
        First dictionary mapping Variable to numpy array (or a numpy matrix).
    dict2: Dict[Variable, np.ndarray | np.matrix]
        Second dictionary mapping Variable to numpy array (or a numpy matrix).

    Returns
    -------
    Dict[Variable, np.ndarray | np.matrix]
        A new dictionary containing all keys from d1 and d2.
        If a key appears in both, its value is d1[key] + d2[key].



    """
    merged_dict = {k: v.copy() for k, v in dict1.items()}

    for k, v in dict2.items():
        merged_dict[k] = merged_dict.get(k, 0) + v.copy()
        if isinstance(merged_dict[k], np.matrix):
            merged_dict[k] = np.asarray(merged_dict[k])
    return merged_dict


def prune_dict(my_dict: dict) -> dict:
    """
    Given a dictionary mapping Variable to a numpy array
    (or a numpy matrix), return a new dictionary containing only
    those items whose array has at least one non-zero elements.
    That is, remove any entries whose array are entirely zero.

    Parameters
    ----------
    my_dict: Dict[Variable, np.ndarray | np.matrix]
        A dictionary mapping variables to their corresponding arrays.

    Returns
    -------
    Dict[Variable, np.ndarray | np.matrix]
        A filtered dictionary with only the entries whose arrays
        contain at least one non-zero values.

    """
    pruned_dict = dict()
    for k, v in my_dict.items():
        try:
            has_nonzero = not np.allclose(v, 0, atol=1e-9)
        except ValueError:
            has_nonzero = v.count_nonzero()
        if has_nonzero:
            pruned_dict[k] = v
    return pruned_dict

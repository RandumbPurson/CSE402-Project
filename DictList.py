import copy
from typing import Callable, Hashable, Any, Iterable, Literal
import numpy as np

class DictArr:
    """A dictionary where each key corresponds to a static array

    Accessing: There are 3 main ways to access content
        - mydictArr["key"]: returns full array corresponding to "key"
        - mydictArr[["key1", "key2"]]: returns tuple of arrays corresponding to the list of keys (eg; "key1" and "key2")
        - mydictArr["key", pos]: returns element at position `pos` in array "key". "key" can also be a list of keys as above
    """
    def __init__(self, init_dict: dict={}):
        """Params:
            - init_dict: a dictionary whose values are all iterables
        """
        for key, val in init_dict.items():
            try:
                iter(val)
            except:
                raise ValueError(f"The value of {key} in init_dict is not iterable!")
        self.dict = copy.deepcopy(init_dict)
        self.items = self.dict.items
        self.keys = self.dict.keys
        self.values = self.dict.values

    def __repr__(self):
        return f"DictList({self.dict})"

    def __str__(self):
        newlinestr = "".join([
            f"\n {key}: {val}" for key, val in self.dict.items()
        ])
        return f"DictList{{{newlinestr}\n}}"

    def __iter__(self):
        return self.dict

    def __getitem__(self, idx: list | Hashable | tuple[list | Hashable, int]):
        if isinstance(idx, list):
            return tuple(self.dict[key] for key in idx) if len(idx) > 1 else self.dict[idx[0]]
            
        if idx in self.dict:
            return self.dict[idx]
            
        if isinstance(idx, tuple):
            key, pos = idx
            return self.__getitem__(key)[pos]
        
        
        raise ValueError(f"Index {idx} not found!")

class DictList(DictArr):
    """A dictionary where each key corresponds to a dynamic list
        > For info on accessing using bracket notation see DictArr
    """
    def _ensurekey(self, key: Hashable):
        if key not in self.dict:
            self.dict[key] = []
    
    def append(self, key: Hashable, val: Any):
        """Append `val` to the `key` list
            - key: The key indicating which list to append to
            - val: The value to append
        """
        self._ensurekey(key)
        self.dict[key].append(val)

    def extend(self, key: Hashable, val: Iterable):
        """Place the contents of the iterable `val` in the `key` list
            - key: The key indicating the destination list
            - val: The source iterable
        """
        self._ensurekey(key)
        try:
            iter(val)
        except:
            raise ValueError(f"The value is not iterable: {val}")

        self.dict[key].extend(val)

    def add(self, key: Hashable, val: Any):
        """A helper function that extends if `val` is iterable and appends otherwise
            - key: The key indicating the list to be added to
            - val: A value to add to the list
        """
        try:
            iter(val)
            self.extend(key, val)
        except:
            self.append(key, val)

    def _join_other(self, join_fn: Callable, other: dict, keyset):
        data = []
        for key in keyset:
            if not (isinstance(key, Hashable) and key in other.keys()):
                raise KeyError(f"'{key}' is not a valid key in {other}")
            join_fn(key, other[key])
            data.append(other[key])
        return data

    def merge(
            self,
            other: dict,
            keys: None | Hashable | list[Hashable] = None,
            mode: Literal["append", "add"] = "add"
        ):
        """Merge another dictionary or DictArr into this DictList
            - other: The dictionary to merge
            - keys(None): The keys to merge from `other`, expanding if possible 
                - <None>: Merge all keys
                - <hashable>: Merge this key
                - <list<hashable>>: Merge keys in list
            - akeys(None): The keys to merge from `other`, always only appending
                - <None>: Ignore
                - <hashable>: Append merge this key
                - <list<hashable>>: Append merge keys in list

            Returns: Returns a tuple of whatever arrays/values were merged in.
        """
        join_fn = {
            "append": self.append,
            "add": self.add
        }[mode]

        if keys is None:
            keys = list(other.keys())

        if not isinstance(keys, list) and isinstance(keys, Hashable):
            keys = [keys]

        if isinstance(keys, list):
            data = self._join_other(join_fn, other, keys)
            if len(data) > 1:
                return tuple(data)
            else:
                return data[0]

    def as_np(self) -> DictArr:
        """Convert to static DictArr where lists become numpy arrays"""
        new_dict = {}
        for key, val in self.dict.items():
            try:
                new_dict[key] = np.array(val)
            except ValueError:
                new_dict[key] = np.array(val, dtype=object)

        return DictArr(new_dict)

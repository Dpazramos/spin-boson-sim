from cross_hamiltonian import CrossHamiltonian
from cross_operator import BitObservable, InputObservable
from cross_state import BitState, InputState
from utils.constants import json_keys

from typing import Union
from qutip import Qobj
import numpy as np

# Auxiliary function to convert simulation elements to a compatible Json format
def to_json_compatible(obj):
    '''
    Converts certain simulation elements to a JSON-compatible format. This is necessary because
    certain objects are not JSON serializable by default. 

    Args:
        obj: Object to convert to JSON-compatible format.

    Returns:
        dict: Dictionary representation of the object in a JSON compatible format.
    '''
    if isinstance(obj, Qobj):
        return {
            "_qutip_Qobj": True,
            "data": obj.data.toarray().tolist(),
            "dims": obj.dims,
            "shape": obj.shape,
            "type": obj.type,
        }

    elif isinstance(obj, complex):
        return {
            "_complex_": True,
            "real": obj.real,
            "imag": obj.imag,
        }

    elif isinstance(obj, Union[BitState, InputState, BitObservable, InputObservable, CrossHamiltonian]):
        return obj.to_dict()

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Auxiliary function to convert from Json to a Qobj
def from_json_compatible(d):
    '''
    Converts certain JSON-compatible elements to their original format. This is necessary because
    certain objects are not JSON serializable by default.

    Args:
        d: Dictionary representation of the object in a JSON compatible format.

    Returns:
        Object: Original object.
    '''
    for key, value in d.items():
        if isinstance(value, dict):
            value_type = value.get("_type")
            if value.get("_qutip_Qobj"):
                d[key] = Qobj(
                    data=value["data"],
                    dims=value["dims"],
                    shape=value["shape"],
                    type=value["type"]
                )
            elif value.get("_complex_"):
                d[key] = complex(value["real"], value["imag"])

            elif value_type in json_keys.keys():
                d[key] = json_keys[value_type].from_dict(value)

            else:
                d[key] = from_json_compatible(value)

            #TODO: add more cases for other classes?

    return d

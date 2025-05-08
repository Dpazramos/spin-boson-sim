# Utility functions for visualisation and data processing
from typing import Union
import numpy as np
import itertools
import pickle
import json
import os

from qutip import Qobj
import matplotlib.colors as mcolors

from utils.constants import (results_io, plot_styles, observable_dict, state_dict)
from cross_hamiltonian import (CrossHamiltonian, Jaynes_Cummings_Hamiltonian)
from cross_operator import (CrossObservable, InputObservable, BitObservable)
from cross_state import CrossState


def read_expect_val(file_path) -> list:
    ''' 
    Read out a '.out' data file for expectation values

    Args:
        file_path (str): Path to the file to read

    Returns:
        list: List of expectation values
    '''
    vals = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and parentheses, then convert to complex number
            val = complex(line.strip().strip('()'))
            vals.append(np.real(val))  # Extract real part if needed

    return vals


def compute_observable(folder_path: str, obs: CrossObservable, keep: list = [], skip_existing: bool = True):
    '''
    Process a folder with various pickled 'results' files to compute and save expectation values for an observable

    Args:
        folder_path (str): Path to the folder containing the results files
        obs (CrossObservable): Observable object for which to compute expectation values
        keep (list): List of strings to demarcate additional files to consider
        skip_existing (bool): Flag to skip existing output files

    '''
    data_files = os.listdir(folder_path)
    pickle_paths = [os.path.join(folder_path, file) for file in data_files if file.endswith('pickle')]

    obs_name = obs.name
    obs_type = obs.type

    for file in pickle_paths:
        file_name = os.path.basename(file)
        base_name = file_name.rsplit('.')[0]

        # Flag to check if the file is recognised 
        found = False
        
        # Initially file name has no suffix
        suffix = ""

        # First, check if the file is in the standard results dictionary
        for res_name, save_name in results_io.items():
            if base_name == res_name:
                print("Considering from standard results:", base_name)
                save_fname = save_name
                found = True
                break

            else:
                for suffix_name in keep:
                    if base_name == res_name + "_" + suffix_name:
                        save_fname = save_name
                        suffix = "_" + suffix_name
                        found = True
                        # print("---------- Identified extra plot:", base_name)
                        # print(f"base name: {save_fname} with suffix {suffix}")
                        break
        
        if not found:
            print(f"Warning - file {base_name} not recognised.")
            continue

        save_fname = f"{save_fname}_{obs_type}{suffix}_{obs_name}.out"
        output_path = os.path.join(folder_path, save_fname)

        # Change the rest of the following code:
    
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping {save_fname} as it already exists.")
            continue

        with open(file, 'rb') as f:
            res = pickle.load(f)

        exp_vals = []
        for subres in res:
            if isinstance(subres, Qobj):
                expec = obs.qt_expect(subres)
            else:
                expec = obs.qk_expect(subres)
                
            exp_vals.append(expec)

        print(f"Saving expectation values for observable '{obs_name}' to {save_fname}")
        np.savetxt(output_path, exp_vals)

    print("\n")


def get_dyn_obs(folder_path: str, obs_name: str, **obs_kwargs) -> CrossObservable:
    ''' 
    Match an observable to the string identifier and simulation data provided in folder_path.

    Args:
        folder_path (str): Path to the folder containing the simulation data
        obs_name (str): Name of the observable to match
        obs_kwargs (dict): Additional keyword arguments for the observable

    Returns:
        CrossObservable: Observable object corresponding to the input parameters
    '''
    params_json = os.path.join(folder_path, "parameters.json")

    if os.path.exists(params_json):
        with open(params_json, 'r') as f:
            params = json.load(f)
    else:
        print(f"Error - parameters.json not found in {folder_path}.")
        return None

    match obs_name:
        case "hamiltonian":
            sim_name = params.get("sim_name", None)
            if sim_name in ["Jaynes-Cummings", "Tavis-Cummings"]:
                observable = Jaynes_Cummings_Hamiltonian.from_dict(params["hamiltonian"])
            else:
                observable = CrossHamiltonian.from_dict(params["hamiltonian"])

        case "initial_state" | "observable":
            if params[obs_name]["_type"] == "BitObservable":
                observable = BitObservable.from_dict(params[obs_name])
            else:
                observable = InputObservable.from_dict(params[obs_name])

        case "auto_mitigated":
            sim_name = params.get("sim_name", None)
            assert(sim_name in ["Tavis-Cummings", "Jaynes-Cummings"]), f"Error - sim_name {sim_name} not valid."
            n_qubits = params.get("n_qubits", None)

            # Try to get the number of qubits from the parameters
            if n_qubits is None:
                try:
                    n_qubits = params["n_atoms"] + 1

                except:
                    try:
                        n_qubits = params["n_sites"] + 1
                    except KeyError:
                        print("Error - n_qubits not found in parameters.")
                        return None
            
            photon_ind = params["hamiltonian"]["photon_ind"] 
            n_excit = len(photon_ind) 

            obs_kwargs = {
                "bitstring": obs_kwargs.get("bitstring", None),
            }

            # Get the valid TC strings to consider
            obs_kwargs["keep_counts"] = valid_TC_strings(n_qubits, n_excit)

            observable = cross_match("bitstring", obs=True, **obs_kwargs)

        case _:
            print(f"Error - observable {obs_name} not recognised.")
            return None

    return observable


def plot_match(fpath: str, obs_name: str, obs_type: str, keep: list = []):
    ''' 
    Parse the folder path based on the input observable data for retrieiving expectation values
    to plot, and additional parameters for the plot.

    Args:
        fpath (str): Path to the file to parse
        obs_name (str): Name of the observable to match
        obs_type (str): Type of observable to match
        keep (list): List of strings to demarcate additional files to consider

    Returns:
        dict: Dictionary containing the parsed data for plotting 
    '''
    fname = os.path.basename(fpath)
    name = fname.split('.')[0]
    print(name)

    keep_mapped = list(map(lambda x: f"{x}_{obs_name}", keep))

    # Initialise the results dictionary
    results = {}

    # Flag to check if the file is recognised
    found = False
    from_keep = False

    for data_name, (plot_type, ls, colour) in plot_styles.items():
        data_name_full = data_name + "_" + obs_type
        if name.startswith(data_name_full):
            # Get the full suffix of the name
            full_suffix = name[len(data_name_full) + 1:]

            if full_suffix == obs_name:
                found = True
                break

            elif full_suffix in keep_mapped:
                found = True
                from_keep = True
                break
            
    if not found:
        # print(f"Warning - data for file {name} not considered.")
        return None
    
    plot_suffix = f" ({full_suffix})"
    plot_name = plot_type + plot_suffix
    linestyle = ls

    vals = read_expect_val(fpath)
        
    results["plot_name"] = plot_name
    results["plot_type"] = plot_type
    results["vals"] = np.array(vals)
    results["linestyle"] = linestyle
    results["suffix"] = full_suffix
    results["keep"] = from_keep
    results["colour"] = colour
    
    return results


# Auxiliary function to get a similar colour to input
def get_similar_colour(color: str, variation=0.3):
    '''
    Get a similar colour to the input colour with a small variation

    Args:
        color (str): Colour to modify
        variation (float): Variation factor for the colour

    Returns:
        np.array: Modified RGB colour
    '''
    # Convert to RGB
    rgb = np.array(mcolors.to_rgb(color))
    print("Old rgb:", rgb)

    # Add a small variaiton to each component
    new_rgb = rgb + np.random.uniform(-variation, variation, size=3)

    # Clip the values to the range [0, 1]
    new_rgb = np.clip(new_rgb, 0, 1)

    return new_rgb


def simulation_match(parameters):
    '''
    Match the input parameters dictionary to a corresponding simulation type, 
    and extract the relevant parameters.

    Args:
        parameters (dict): Dictionary of input parameters

    Returns:
        dict: Dictionary of parsed general simulation parameters
        dict: Dictionary of additional parsed parameters    
    '''
    sim_name = parameters.get("sim_name", None)

    if sim_name is None:
        print(f"Error - No simulation name found in {parameters}.")
        return None, None
    else:
        if (sim_name == "Tavis-Cummings") or (sim_name == "Jaynes-Cummings"):
            g = parameters.get('g', None)
            omega = parameters.get('omega', None)
            n_atoms = parameters.get('n_atoms', None)

            if (omega is None) or (g is None) or (n_atoms is None):
                print(f"Error - insufficient parameters (g {g}, omega {omega}, n_atoms {n_atoms}).")
                return None, None
            
            title = f"n_atoms={parameters['n_atoms']}, omega={omega}, g={g}"
            
            sim_parsed = {
                "combination": str((omega, g)), 
                "n_qubits": (n_atoms + 1),
                "title": title,
                "sim_name": sim_name,
            }

            extra_params = {
                "g": g,
                "omega": omega,
                "n_atoms": n_atoms,
            }

        elif sim_name.startswith("Heisenberg Spin"):
            J = parameters.get('J', None)
            h = parameters.get('h', None)
            n_sites = parameters.get('n_sites', None)

            # TODO: parse title type

            if (h is None) or (J is None) or (n_sites is None):
                print(f"Error - insufficient parameters (J {J}, h {h}, n_sites {n_sites}).")
                return None, None
            
            title = f"n_sites={n_sites}, J={J}, h={h}"

            sim_parsed = {
                "combination": str((J, h)),
                "n_qubits": (n_sites + 1),
                "title": title,
                "sim_name": sim_name,
            }

            extra_params = {
                "J": J,
                "h": h,
                "n_sites": n_sites,
            }

        else:
            print(f"\n**** Warning - Parameter parsing for '{sim_name}' not implemented. ****\n")
            return None, None

        print(f"Parsing {sim_name} successful.")
        return sim_parsed, extra_params
    

def valid_TC_strings(n_atoms: int, n_excitations: int = 1) -> list:
    '''
    Get the valid bitstrings for a given number of qubits and excitations in the Tavis-Cummings model

    Args:
        n_atoms (int): Number of atoms in the system
        n_excitations (int): Number of excitations in the system

    Returns:
        list: List of valid bitstrings for the parametrised Tavis-Cummings model
    '''
    # Define the valid bitstring permutations for the Tavis-Cummings model with one excitation
    n = n_atoms
    m = n_excitations

    if m > n:
        raise ValueError("Number of excitations cannot exceed the number of qubits.")
    
    # Generate all indices combinations for '1's
    indices_combinations = itertools.combinations(range(n), m)
    bitstrings = []
    
    # Gets all combinations with exactly m '1's, and then flips all bits except the leftmost digit
    for indices in indices_combinations:
        bitstring = ['0'] * n

        for index in indices:
            bitstring[index] = '1'

        flipped_bitstring = [bitstring[0]] + ['1' if bit == '0' else '0' for bit in bitstring[1:]]
        bitstrings.append(''.join(flipped_bitstring))

    return bitstrings


def cross_match(key_cross: str, obs: bool, **kwargs) -> Union[CrossObservable, CrossState]:
    '''
    Match the corresponding key to the CrossObject and return the object with the input parameters

    Args:
        key_cross (str): Key to match
        obs (bool): Flag to match observable or state
        **kwargs: Additional keyword arguments for the CrossObject

    Returns:
        Union[CrossObservable, CrossState]: CrossObject corresponding to the input key
    '''
    cross_dict = observable_dict if obs else state_dict

    if key_cross in cross_dict.keys():
        cross = cross_dict[key_cross]
    else:
        raise ValueError(f"Cross-object {key_cross} not recognised (obs {obs})")

    if cross is not None:
        return cross(**kwargs)
    else:
        raise ValueError(f"Cross-key {key_cross} not found (obs {obs})")

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from visualise import get_output_path, read_out
import matplotlib.pyplot as plt
import os
import numpy as np

# First choose the folder corresponding to the simulation to be visualised
parent_directory = "..." # Insert the name of the folder containing the simulation results here
full_path = get_output_path(parent_directory)

# Define color for each evolution method
colors = {
    "Exact evolution": "blue",
    "ISL evolution": "green",
    "Exact Trotterized evolution": "red",
    "Noisy Trotterized evolution": "orange",
    "Other": "black",
}

# Simulation parameters
dt = 0.01
t = None # Initialise as None to be defined later
steps = 20

subdirectories = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]

for folder_name in subdirectories:
    print(f"\n\nFolder: {folder_name}")
    folder_path = os.path.join(parent_directory, folder_name)

    # Get all files in the output directory
    path_out = get_output_path(folder_path)
    out_files = os.listdir(path_out)

    # Get .out files
    out_paths = [os.path.join(path_out, file) for file in out_files if file.endswith('.out')]

    # Initialise a plot for the current folder
    plt.figure()

    # Plot each file
    for file in out_paths:
        file_name = os.path.basename(file)
        match file_name:
            case "qutip_probabilities.out": 
                plot_name = "Exact evolution"
            case "isl_probabilities.out": 
                plot_name = "ISL evolution"
            case "Exact_trotter_probabilities.out":
                plot_name = "Exact Trotterized evolution"
            case "Noisy_trotter_probabilities.out":
                plot_name = "Noisy Trotterized evolution"
            case _:
                plot_name = "Other"
        probs = read_out(file)
        
        # Optional: print plot_name for reference
        # print(plot_name)

        # Define the time steps
        t = np.linspace(0, steps * dt, steps + 1)

        # Exact evolution - higher resolution, hence t must be rescaled
        if plot_name == "Exact evolution":
            t = np.linspace(t[0], t[-1], len(probs))
        # Interpolating observable
        # If unequal lengths, then interpolate the probabilities to adjust to the time steps
        elif len(probs) != len(t):
            xp = np.linspace(t[0], t[-1], len(probs))
            probs = np.interp(t, xp, probs)
        
        plt.plot(t, probs, color=colors[plot_name], label=plot_name)

    plt.xlabel('time step (s)')
    plt.ylabel('Probability of initial state')
    plt.title('Different evolution methods')
    plt.grid(True)
    plt.legend()
    plt.show()



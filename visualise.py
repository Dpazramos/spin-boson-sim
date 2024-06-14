import matplotlib.pyplot as plt
import os
import numpy as np

# Get directory to the output folder
path_out = os.path.join(os.getcwd(), 'output')
out_files = os.listdir(path_out)

# Define function to read_out .out files
def read_out(file_path):
    probs = []
    with open(file_path, 'r') as file:
        for line in file:
            prob = float(line.strip())
            probs.append(prob)
    return probs

# Define time interval (user-input)
t = np.linspace(0, 0.01, 20+1)  # time-steps

# Define colors for each plot entry
colors = {
    "Exact evolution": "blue",
    "ISL evolution": "green",
    "Trotterized evolution": "red"
}


# Get .out files
out_paths = []
for file in out_files:
    if file.endswith('.out'):
        file_output = os.path.join(path_out, file)
        out_paths.append(file_output)

# Plot each file
for file in out_paths:
    file_name = os.path.basename(file)
    match file_name:
        case "qutip_probabilities.out": 
            plot_name = "Exact evolution"
        case "isl_probabilities.out": 
            plot_name = "ISL evolution"
        case "trotter_probabilities.out":
            plot_name = "Trotterized evolution"
            
    probs = read_out(file)

    if len(probs) > len(t):
        xp = np.linspace(0, 0.01, len(probs))
        probs = np.interp(t, xp, probs)
        print("interpolating probabilities")

    plt.plot(t, probs, color=colors[plot_name], label=plot_name)

plt.xlabel('time step (s)')
plt.ylabel('Probability of initial state')
plt.title('Different evolution methods')
plt.grid(True)
plt.legend()
plt.show()
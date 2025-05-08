#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --mem=50G
#SBATCH --output=3q_16384_TCM_(0-3)_base_%a.out
#SBATCH --job-name=3q_16384_TCM_(0-3)_base
#SBATCH --cpus-per-task=10

# Create an array of N jobs with index values 0, ..., N
#SBATCH --array=0-3
# (usually 0-3 for TCM, 0-4 for Heisenberg models)
# (An alternative could be having specific indices, e.g., array=2,5)

# Line for testing standard shell commands
echo "Hello $USER! You are using the node $HOSTNAME.(time $(date))"

# Print job step id
srun echo "I am array task number $SLURM_ARRAY_TASK_ID"

# Simulation parameters
run_type=TCM
n_qubits=3

# simuls=('exact_trotter' 'noisy_trotter' 'zne' 'noisy_isl')
simuls=('noisy_trotter', 'exact_trotter')
# simuls=('zne')
# simuls=('noisy_isl')
tol=1e-2
initial_state=ones
observable=manual_ones
gen_shots=16384

# Define the label for the output file
sub_data_label="${gen_shots}"
# data_label="${tol}_${sub_data_label}"
data_label="${sub_data_label}"
label="${observable}_${sub_data_label}_shots_${SLURM_ARRAY_TASK_ID}"

# First cas estatement to determine the parameters to use
case $run_type in 
    "HS")       # OLD CASE
        # Case matching for specific array (hard-coded)
        case $SLURM_ARRAY_TASK_ID in
            0) h=(-20 -20 -20); J=(-20 -20 -2);;
            1) h=(-20 -20 -20); J=(-2 -2 -20);;
            2) h=(-20 -20 -20); J=(-20 -20 -20);;
            3) h=(-20 -20 -20); J=(-2 -20 -40);;
            4) h=(0 0 -2); J=(-10 10 0);;
        esac
        ;;

    "QHM") 
        case $SLURM_ARRAY_TASK_ID in
            0) h=(0 0 -2); J=(-10 10 0);;   # Main (XY)
            1) h=(0 0 -2); J=(-10 10 20);;   # (XYZ)
            2) h=(0 0 -2); J=(10  10 10);;  # (XXX)
            3) h=(0 0 -2); J=(10 10 0);;    # (XX)
            4) h=(0 0 -2); J=(0 10 0);;   # (Ising)
        esac
        ;;

    "TCM")
        # Case matching for specific array (hard-coded)
        case $SLURM_ARRAY_TASK_ID in
            0) g=10; omega=1;;
            1) g=5; omega=1;;
            2) g=20; omega=1;;
            3) g=1; omega=1;;
            4) g=10; omega=10;;
            5) g=10; omega=0.5;;
            6) g=10; omega=5;;
        esac
        ;;
esac

# **** Define static simulation parameters ****
# Time domain
delta_t=0.01
n_steps=40

# Additonaly parameters for algorithms
trot_shots=$gen_shots
isl_shots=$gen_shots
zne_shots=$gen_shots
zne_avg=10

# Join the elements of simuls array into a comma-separated string
IFS=','; simuls_str="${simuls[*]}"; IFS=' '

script_path="ext-spin-bos-sim/src/args_sim.py"

# Echo the parameters
echo "Simulation-type: $run_type"
echo "Chosen algorithms: $simuls_str"
srun echo "I have the general parameters: delta_t: $delta_t, n_steps: $n_steps,"
srun echo "trot_shots: $trot_shots, isl_shots: $isl_shots, tol: $tol, zne_shots: $zne_shots"
echo "Running script at path: $script_path"

# *** Run the simulation *** #
# Heisenberg-Spin (Quantum Heisenberg Model)
if  [ "$run_type" == "HS" ] || [ "$run_type" == "QHM" ]; then
    n_sites=$n_qubits
    srun echo "Sim-dependent parameters: n_sites: $n_sites, h: (${h[@]}), J: (${J[@]})" 
    srun python3 "$script_path" "HS" \
    "$n_sites" "${h[@]}" "${J[@]}" \
    "$delta_t" "$n_steps" "$trot_shots" "$isl_shots" "$tol" "$zne_shots" "$initial_state" "$observable" \
    -za "$zne_avg" -sv -l "$label" -sim "$simuls_str"  -dl "$data_label" -est -o "output"
# Tavis-Cummings
else
    n_atoms=$((n_qubits-1))
    srun echo "Sim-dependent parameters: n_atom(s): $n_atoms, omega: $omega, g: $g"
    srun python3 "$script_path" "TC" \
    "$n_atoms" "$omega" "$g" \
    "$delta_t" "$n_steps" "$trot_shots" "$isl_shots" "$tol" "$zne_shots" "$initial_state" "$observable" \
    -za "$zne_avg" -sv -l "$label" -sim "$simuls_str" -dl "$data_label" -est -o "output"
fi

# Indicate job completion
srun echo "Task $SLURM_ARRAY_TASK_ID successfully completed. (time $(date))"

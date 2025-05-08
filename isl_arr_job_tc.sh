#!/bin/bash -l
#SBATCH --time=40:00:00
#SBATCH --mem=44G
#SBATCH --output=1e-2_4q_16384_noisy_isl_TC_%a.out
#SBATCH --job-name=1e-2_4q_16384_noisy_isl_TC
#SBATCH --cpus-per-task=18

# Create an array of N jobs with index values 0, ..., N
#SBATCH --array=0-0
# (An alternative could be having specific indices, e.g., array=2,5)

# Line for testing standard shell commands
echo "Hello $USER! You are using the node $HOSTNAME.(time $(date))"

# Print job step id
srun echo "I am array task number $SLURM_ARRAY_TASK_ID"
srun echo

# **** Define simulation parameters ****
n_atoms=3
shots=16384
tol=1e-2
noiseless_sim=aer

simuls=('noisy_isl')
omega=1
g=10
delta_t=0.01
n_steps=40
trot_shots=$shots
isl_shots=$shots
zne_shots=$shots
initial_state=ones
observable=ones

label=exact_isl_shots_${shots}_tol_${tol}

# Join the elements of simuls array into a comma-separated string
IFS=','; simuls_str="${simuls[*]}"; IFS=' '

# Construct the path
sim_path="arg_tc.py"
base_path="ext-spin-bos-sim/src"
dynamic_path="$base_path/$sim_path"

# Print the parameters
srun echo "Chosen simulations: ${simuls[@]}"
srun echo "I have the sim. parameters omega: $omega, g: $g, atom(s): $n_atoms"
srun echo "n_steps: $n_steps, delta_t: $delta_t, trot_shots: $trot_shots, \ 
        isl_shots: $isl_shots, tol: $tol, zne_shots: $zne_shots"
srun echo "Running script at path: $dynamic_path"

# Run the job
srun python3 "$dynamic_path" \
    "$n_atoms" "$omega" "$g" \
    "$delta_t" "$n_steps" "$trot_shots" "$isl_shots" "$tol" "$zne_shots" "$initial_state" "$observable" \
    -sv -l "$label" -sim "$simuls_str" -ns "$noiseless_sim" -o output
	
# Indicate job completion
srun echo "Task $SLURM_ARRAY_TASK_ID successfully completed. (time $(date))"

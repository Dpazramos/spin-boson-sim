#!/bin/bash -l
#SBATCH --time=30:00:00
#SBATCH --mem=40G
#SBATCH --output=16384_exact_isl_halfstep_1_n_atoms.out
#SBATCH --job-name=exact_isl_halfstep
#SBATCH --cpus-per-task=10

# Line for testing standard shell commands
echo "Hello $USER! You are using the node $HOSTNAME.(time $(date))"
echo
# **** Define simulation parameters ****
n_atoms=2
simuls=('exact_isl')
omega=1
g=10
delta_t=0.005
n_steps=80
trot_shots=1024
isl_shots=16384
tol=0.01
zne_shots=1000
initial_state=ones
observable=ones
label=exact_isl_halfstep_$isl_shots

# Join the elements of simuls array into a comma-separated string
IFS=','; simuls_str="${simuls[*]}"; IFS=' '

# Construct the path
sim_path="arg_tc.py"
base_path="ext-spin-bos-sim/src"
dynamic_path="$base_path/$sim_path"

# Print the parameters
echo "Chosen simulations: ${simuls[@]}"
echo "I have the sim. parameters omega: $omega, g: $g, atom(s): $n_atoms"
srun echo "I have the parameters: n_steps: $n_steps, delta_t: $delta_t, trot_shots: $trot_shots, \
isl_shots: $isl_shots, tol: $tol, zne_shots: $zne_shots"
echo "Running script at path: $dynamic_path"
echo

# Run the job
srun python3 "$dynamic_path" \
    "$n_atoms" "$omega" "$g" \
    "$delta_t" "$n_steps" "$trot_shots" "$isl_shots" "$tol" "$zne_shots" "$initial_state" "$observable" \
    -sv -l "$label" -sim "$simuls_str" -o output
	
# Indicate job completion
srun echo "Task $SLURM_ARRAY_TASK_ID successfully completed. (time $(date))"

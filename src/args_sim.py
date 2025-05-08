from utils.args import gen_args, hs_args, tc_args
from sim import HS_Simulator, TC_Simulator

from argparse import ArgumentParser

def main():
    # Create an argument parser
    parser = ArgumentParser(prog="Spin-boson Simulator", description="Simulator for Spin-boson models")

    # Add argument to choose model type
    parser.add_argument("model_type", choices=["HS", "TC"], help="Choose model type ('HS': Heisenberg-Spin, 'TC': Tavis-Cummings)")

    # Temporarily parse to get the model type
    args, remaining_args = parser.parse_known_args()
    model_type = args.model_type

    # Depending on the model type, instantiate the simulator and add specific arguments
    if model_type == "HS":
        print("\n****Instantiating Simulator for the Heisenberg-Spin Model****\n\n")
        sim = HS_Simulator()
        hs_args(parser)

    elif model_type == "TC":
        print("\n****Instantiating Simulator for the Tavis-Cummings Model****\n\n")
        sim = TC_Simulator()
        tc_args(parser)
        print("Instantiation complete.\n\n")

    # Add the common arguments
    gen_args(parser)

    # Pass the full argument list, and parse
    remaining_args = [model_type] + remaining_args
    args = parser.parse_args(remaining_args)

    # Get the simulations into the form of a list
    simulations = [sim.strip() for sim in args.simulations.split(",") if len(sim.strip()) > 0]

    # Parse common arguments
    arguments = {
    'delta_t': args.delta_t,
    'n_steps': args.n_steps,
    'trot_shots': args.trot_shots,
    'isl_shots': args.isl_shots,
    'tolerance': args.tolerance,
    'zne_shots': args.zne_shots,
    'zne_avg': args.zne_avg,
    'initial_state': args.initial_state,
    'observable': args.observable,
    'estimator': args.estimator,
    'debug': args.debug,
    'save': args.save,
    'noiseless_simulator': args.noiseless_simulator,
    'output': args.output,
    'data_label': args.data_label,
    'label': args.label,
    'simulations': simulations,
    }

    # Get simulation-specific arguments
    if model_type == "HS":
        aux_arguments = {
            'n_sites': args.n_sites,
            'h': args.h,
            'J': args.J
        }

    elif model_type == "TC":
        aomega = args.omega if args.aomega == -1 else args.aomega
        aux_arguments = {
            'n_atoms': args.n_atoms,
            'omega': args.omega,
            'aomega': aomega,
            'g': args.g
        }
    
    arguments.update(aux_arguments)

    # Call function to pass-in arguments
    print("****Setting up Simulations****\n")
    sim.setup_simulation(**arguments)

    # Run the simulation
    print("****Running Simulations****\n")
    sim.run_simulations()
    
    print("****Simulations Complete****\n")

if __name__ == "__main__":
    main()

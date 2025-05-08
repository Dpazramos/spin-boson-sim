from argparse import ArgumentParser

from utils.args import gen_args, hs_args
from sim import HS_Simulator

def main():
    # Instantiate the simulator
    print("\n****Instantiating Simulator for the Heisenberg-Spin Model****\n\n")
    sim = HS_Simulator()

    prog_name, description = sim.get_description()

    # Create an argument parser
    parser = ArgumentParser(prog=prog_name, description=description)

    # Add arguments - and parse
    hs_args(parser)
    gen_args(parser)
    args = parser.parse_args()

    # Get the simulations in the form of a list
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
        'debug': args.debug,
        'save': args.save,
        'output': args.output,
        'data_label': args.data_label,
        'label': args.label,
        'simulations': simulations,
    }

    # Get simulation (HS)-specific arguments
    aux_arguments = {
        'n_sites': args.n_sites,
        'h': args.h,
        'J': args.J,
    }
    arguments.update(aux_arguments)

    # Call function to pass-in arguments
    print("****Setting up Simulations****\n")
    sim.setup_simulation(**arguments)

    # Run the simulation
    print("****Running Simulations****\n")
    sim.run_simulations()
    print("\nScript complete.\n")

# Mark entry point for the script
if __name__ == '__main__':
    main()
 
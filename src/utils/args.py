# Define arguments to pass into the simulator

# General simulation arguments
def gen_args(parser) -> None:
    '''
    General arguments for the simulator
    '''
    parser.add_argument("delta_t", help="size of time step", type=float, default=0.01)
    parser.add_argument("n_steps", help="number of time steps", type=int, default=20)
    parser.add_argument("trot_shots", help="shots per circuit evaluation in ISL", type=int, default=1024)
    parser.add_argument("isl_shots", help="shots per circuit evaluation in ISL", type=int, default=1024)
    parser.add_argument("tolerance"  , help="tolerance for ISL", type=float, default=1e-2)
    parser.add_argument("zne_shots", help="shots per circuit evaluation in ZNE", type=int, default=16384)
    parser.add_argument("initial_state", help="initial state of the system", type=str, default="ones")
    parser.add_argument("observable", help="observable to measure", type=str, default="ones")
    parser.add_argument("-est", "--estimator", help="toggle estimator mode", action="store_true", default=True)
    parser.add_argument("-za", "--zne_avg", help="shots per circuit evaluation in ZNE", type=int, default=1)
    parser.add_argument("-d", "--debug", help="toggle debug mode", action="store_true", default=False)
    parser.add_argument("-p", "--parallel", help="toggle parallel execution", action="store_true", default=False)
    parser.add_argument("-sv", "--save", help="save results to file", action="store_true", default=False)
    parser.add_argument("-o", "--output", help="name of main output folder", default="output")
    parser.add_argument("-l", "--label", type=str, default="", help="label for output subfolder")
    parser.add_argument("-dl", "--data_label", type=str, default="", help="label for data files")
    parser.add_argument("-ns", "--noiseless_simulator", help="noiseless simulator", type=str, default='aer')
    parser.add_argument("-sim", "--simulations", help="comma-separated list of simulations to run (e.g., 'exact, noisy_trotter, exact_isl, zne')", type=str, default="")

def tc_args(parser) -> None:
    '''
    Arguments for the Tavis-Cummings model
    '''
    parser.add_argument("n_atoms", help="number of atoms", type=int, default=1)
    parser.add_argument("omega", help="cavity frequency", type=float, default=10)
    parser.add_argument("-aomega", help="atom frequency", type=float, default=-1)
    parser.add_argument("g", help="interaction strength", type=float, default=1)

def hs_args(parser) -> None:
    '''
    Arguments for the Heisenberg Spin model
    '''
    parser.add_argument("n_sites", help="number of sites", type=int, default=1)
    parser.add_argument("h", help="magnetic field strengths", type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument("J", help="interaction strengths", type=float, nargs=3, default=[1, 1, 1])

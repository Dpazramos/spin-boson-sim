from sim import TC_Simulator, HS_Simulator    

def main():
    system = 'tc'

    print(f"Running simulation for system: {system}.\n")
    
    match system:
        case 'tc':
            sims = ['noisy_isl']

            # Define arguments
            params = {
                'n_atoms': 2,
                'initial_state': 'ones',
                'observable': 'manual_ones',
                'delta_t': 0.01,
                'n_steps': 10,
                'trot_shots': 16384,
                'isl_shots': 1024,
                'zne_shots': 100,
                'zne_avg': 10,
                'omega': 1,
                'aomega':1,
                'g': 10,
                'tolerance': 1e-2,
                'label': 'test',
                'data_label': '',
                'simulations': sims,
                'save': False,
                'debug': False,
                # 'noiseless_simulator': 'aer'
                # 'parallel': False,
            }
        
            tc_sim = TC_Simulator()
            tc_sim.run_simulations(**params)
            

        case 'hs':
<<<<<<< HEAD
            sims = ['noisy_isl',]
=======
            sims = ['exact_isl',]
>>>>>>> 444bcd0fbb87b87a2263651f6323d9f4d9121cb9
            params = {
                'n_sites': 3,
                'initial_state': 'ones',
                'observable': 'manual_ones',
                'delta_t': 0.01,
                'n_steps': 10,
                'J': [-10, 10, 0],
                'h': [0, 0, -2],
                'label': 'test',
                'data_label': '',
                'simulations': sims,
                'save': False,
                'debug': False,
                'trot_shots': 4096,
                'isl_shots': 4096,
            }

            hs_sim = HS_Simulator()
            hs_sim.run_simulations(**params)

        case _:
            print(f"System {system} not recognised.")

    print("\nScript complete.\n")


if __name__ == '__main__':
    main()

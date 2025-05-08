#coding: utf-8
#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Tuple, Union
import datetime
import pickle
import time
import json
import os

# from concurrent.futures import ProcessPoolExecutor

from qiskit.providers.fake_provider import FakeNairobi
from qiskit.circuit.library import PauliEvolutionGate
from isl.recompilers import ISLRecompiler, ISLConfig
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, StatevectorSimulator, QasmSimulator
from qiskit_aer.noise import NoiseModel
from mitiq import zne
import qutip as qt
import numpy as np

from cross_hamiltonian import (CrossHamiltonian, Tavis_Cummings_Hamiltonian, Heisenberg_Hamiltonian)
from utils.qindex import get_layout_qubits, truncate_to_active_qubits, update_coupling_map
from utils.json_conversions import from_json_compatible
from utils.json_conversions import to_json_compatible
from utils.visualise import valid_TC_strings
from cross_operator import CrossObservable
from utils.visualise import cross_match
from cross_state import CrossState


##########################
### Abstract Simulator ###
##########################
class Spin_Boson_Simulator(ABC):
    """
    Class of Simulator objects for running simulations of spin-boson models
    """
    def __init__(self):
        self._set_up = False    # Indicate if the setup has been perfomred
        self.label = False      # Indicate if a label for the subfolder is provided
        self._dict = None

    #############
    ### SETUP ###
    #############
    # Procedure to construct output folder
    def set_output_path(self):
        if not self._save:
            print("Warning: Not creating path - saving is disabled.\n")
            return

        mainparams, subparams = self.get_params()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subfolder_name = f"{mainparams}_{timestamp}_({subparams}_)"

        # No time domain information is included in the subfolder name (for brevity purposes)

        # Add a label if provided
        if len(self._label) > 0:
            subfolder_name = f"{self._label}_{subfolder_name}"

        # Store folder one path up
        output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', self._output_folder))
        print(f"Saving data to {output_folder}")
        self._output_folder = os.path.join(output_folder, subfolder_name)

        try:
            os.makedirs(self._output_folder)
        except:
            print(f"Warning - output subfolder already exists {self._output_folder}")


    # Function to save parameters as metadata to json file
    def save_parameters(self):
        if not self._save:
            print("Warning: Not saving parameters - saving is disabled.\n")
            return

        dict_args = from_json_compatible(self.dict_args)

        params_file = os.path.join(self._output_folder, 'parameters.json')
        print(f"parameters: {dict_args}")
        with open(params_file, 'w') as f:
            json.dump(dict_args, f, indent=1, default=to_json_compatible)
        print(f"Parameters saved to {params_file}.\n")

    # Method to setup simulation
    def setup_simulation(self, **kwargs):
        # ***** Get the kwargs *****
        # Time domain
        self._n_steps = kwargs.get("n_steps", 20)
        self._delta_t = kwargs.get("delta_t", 0.01)
        T = self._delta_t * self._n_steps
        self._t = np.linspace(0, T, self._n_steps + 1)
        self._t_steps = self._t.size

        # Simulation parameters
        self._simulations = kwargs.get("simulations", [])
        self.trot_shots = kwargs.get("trot_shots", 1024)
        self.isl_shots = kwargs.get("isl_shots", 1024) if self.has_method("isl") else None
        self.tol = kwargs.get("tolerance", 1e-2) if self.has_method("isl") else None
        self.zne_shots = kwargs.get("zne_shots", 16384) if self.has_method("zne") else None
        self.zne_avg = kwargs.get("zne_avg", 1) if self.has_method("zne") else None 

        # Output specifications
        self._save = kwargs.get("save", True)
        self._debug = kwargs.get("debug", False)
        # self._parallel = kwargs.get("parallel", False)
        self._output_folder = kwargs.get("output", "output") if self._save else None
        self._label = kwargs.get("label", "") if self._save else None
        data_label = kwargs.get("data_label", "")
        self._data_label = f"_{data_label}" if len(data_label) > 0 else ""

        if self._debug:
            print("Warning: Debugging mode has been toggled.\n")

        # Run context-dependent setup procedure
        self.aux_setup_simulation(**kwargs)

        # ---- Miscellaneous ---- #
        # Encoding
        self._encoding = kwargs.get("encoding", "graycode")
        

        # Construct the simulation-dependent Hamiltonian variables
        self.H = self.get_hamiltonian()
        self._h_qt = self.H.full_hamiltonian()              # Uses big-endian order
        h_q = self.H.map_to_qubits(self._encoding)          # Uses little-endian order


        # Initial state
        initial_state = kwargs.get("initial_state", "ones")
        self._initial_state = self.get_cross(initial_state, obs=False)


        # Observable
        observable = kwargs.get("observable", "ones")
        self._observable = self.get_cross(observable)

        self._estimator = kwargs.get("estimator", True) if not self._observable.manual else False
        self._results_suffix = "_estimator" if self._estimator else ""

        # Set the path for saving results, and store parameter settings
        self.mainparams, self.subparams = self.get_params()

        if self._debug:
            print("\nDebug Setup:")
            print("Hamiltonian:", self.H,"\n")
            print("Full Hamiltonian:", self._h_qt,"\n")
            print("Mapped Hamiltonian:", h_q,"\n")
            print("\n")
        
        # Store Trotter step = exp(-iHt)(delta_t), H is represented by a SparsePauliOp object
        self.trotter = PauliEvolutionGate(h_q, time=self._t[1])
        self.circuit_list = []

        # Get backends
        self.noise_model = NoiseModel.from_backend(FakeNairobi())  # Note: The noise model is fixed to that of the Nairobi backend
        self.noisy_backend = AerSimulator.from_backend(FakeNairobi(), method='density_matrix')

        # Identify noiseless backend to use
        noiseless_backend = kwargs.get("noiseless_simulator", "aer")
        if noiseless_backend in ["aer", "aer_simulator"]:
            self.exact_backend = AerSimulator(method='density_matrix')
            self.noiseless_backend_name = "aer_simulator"
        elif noiseless_backend in ["qasm", "qasm_simulator"]:
            self.exact_backend = QasmSimulator(method='density_matrix')
            self.noiseless_backend_name = "qasm_simulator"
        elif noiseless_backend in ["sv", "statevector", "statevector_simulator"]:
            self.exact_backend = StatevectorSimulator()
            self.noiseless_backend_name = "statevector_simulator"
        else:
            raise ValueError(f"Invalid noiseless simulator backend: {noiseless_backend}")

        # Flags
        self.trot_list = False # Flag to determine whether Trotter circuits have been computed
        self._set_up = True # Indicate setup completion

        # Set the output folder, and save the parameters in a JSON format
        if self._save:
            self.set_output_path()
            self.save_parameters()


    # Define dictionary of arguments for a simulation run
    @property
    def dict_args(self):
        if self._dict is None:
            common_dict = {
                'sim_name': self.simulation_name,
                'n_steps': self._n_steps,
                'delta_t': self._delta_t,
                'n_qubits': self.n_qubits,
                'encoding': self._encoding,
                'noiseless_simulator': self.noiseless_backend_name,
                'estimator': self._estimator,
                'simulations': self._simulations,
                'isl_shots': self.isl_shots,
                'trot_shots': self.trot_shots,
                'zne_shots': self.zne_shots,
                'zne_avg': self.zne_avg,
                'tolerance': self.tol,
                'observable': self._observable,
                'initial_state': self._initial_state,
                'hamiltonian': self.H,
            }
            all_dict = self.dict_aux_args.copy()
            all_dict.update(common_dict)
            self._dict = all_dict
        return self._dict


    # Simulate the exact time evolution of the system (QuTiP)
    def exact_simulation(self):
        print("Exact simulations start.")
        time_taken = time.time()
        init_state = self._initial_state.qt_state
        exact_result = qt.sesolve(self._h_qt, init_state, self._t, [])
        time_taken  = time.time() - time_taken
        print(f"Exact simulations end. (time taken: {time_taken})\n")

        exact_expect = []
        exact_rho = []
        for state in exact_result.states:
            # Compute the expectation value of the observable acting on the state
            prob = self._observable.qt_expect(state)
            exact_expect.append(prob)
            rho = state * state.dag()
            exact_rho.append(rho)

        if self._save:
            # Save the density matrix at each time step
            with open(self._output_folder + f"/qutip_results{self._data_label}.pickle", 'wb') as f:
                pickle.dump(exact_rho, f, pickle.HIGHEST_PROTOCOL)
            
            # Save the expectation values of the observable at each time step
            np.savetxt(self._output_folder+f"/qutip_{self._observable.type}{self._data_label}_{self._observable.name}.out", exact_expect)

        if self._debug:
                print("\nQuTiP Debug ")
                print("Observable State:", self._observable.name)
                print("Initial state:", self._initial_state.qt_state)
                print("Hamiltonian:", self._h_qt)
                for expect, state in zip(exact_expect, exact_rho):
                    print("Expect. value:", expect)
                    print("State:", state)
                print("End QuTiP debug.\n")

        if self._save:
            np.savetxt(self._output_folder+f"/qutip_{self._observable.type}{self._data_label}_{self._observable.name}.out", exact_expect)


    # Check whether a speciified method is to be run
    def has_method(self, method):
        match method:
            case 'exact':
                return 'exact' in self._simulations
            case 'trotter':
                return ('noisy_trotter' in self._simulations or 'exact_trotter' in self._simulations)
            case 'isl':
                return ('noisy_isl' in self._simulations or 'exact_isl' in self._simulations)
            case 'zne':
                return 'zne' in self._simulations
            case _:
                return False


    # Convert an input string into the corresponding defined 'CrossObservable'/'CrossState' object
    def get_cross(self, name: str, obs : bool = True) -> Union[CrossObservable, CrossState]:
        if set(name).issubset({'0', '1'}) and len(name) != self.n_qubits:
            raise ValueError(f"Invalid length in initial input: n_qubits: {self.n_qubits}, input: {name}.")

        key_cross = name

        # Repeated bit
        if name in ["ones", "zeros"]:
            key_cross = "repbit"
            kwargs = {
                "type": name,
                "n_qubits": self.n_qubits,
            }

        # Generic bitstring
        elif set(name).issubset({'0', '1'}):
            if len(name) == self.n_qubits:
                kwargs = {'bitstring': name}
                key_cross = "bitstring"

        # Manual counts (bitstring) (given in the form of 'manual_....' or 'mitig_...')
        elif name.startswith('manual') or name.startswith('mitig'):
            suffix = name.split('_')[-1]  
            kwargs = {}

            if name.startswith('mitig'):
                if self.simulation_name in ["Tavis-Cummings", "Jaynes-Cummings"]:
                    print("Simulation:", self.simulation_name)
                    keep_counts = valid_TC_strings(self.n_qubits, len(self.H.photon_ind))
                    kwargs['keep_counts'] = keep_counts

                else:
                    print(self.simulation_name)
                    print("Warning: Mitigation is not implemented for this model.\nUsing standard manual counts")
            else:
                print(f"Manual counts chosen (sim {self.simulation_name}, obs {name}).")

            if set(suffix).issubset({'0', '1'}):
                extra_kwargs = {
                    'bitstring': suffix,
                    'manual_counts': True,
                    }
                key_cross = "bitstring"

                kwargs.update(extra_kwargs)

            elif suffix in ["ones", "zeros"]:
                extra_kwargs = {
                    'type': suffix,
                    'n_qubits': self.n_qubits,
                    'manual_counts': True,
                }
                key_cross = "repbit"

                kwargs.update(extra_kwargs)
            else:
                raise ValueError(f"Invalid manual observable name ({name}).")

        # Z spin
        elif name in ["atom_energy",]:
            key_cross = "addspin"

            atom_spins = ["Z" for _ in range(self._n_atoms)]
            kwargs = {
                'axes': ["0"] + atom_spins,
            }

        # Generic Spin
        elif set(name).issubset({'x', 'y', 'z', '0'}):
            key_cross = "addspin"
            kwargs = {
                'axes': list(map(lambda x: x.upper(), name))
            }

        # Repeated spin
        elif name in ["spinZ", "spinX", "spinY"]:
            ax = name[-1]
            key_cross = "repaddspin"
            kwargs = {
                'n_qubits': self.n_qubits,
                'axis': ax,
            }

        # Hamiltonian of the given system
        elif name in ["hamiltonian", "energy"]:
            return self.H

        elif name in ["psiones", "psirest"]:
            key_cross = "psirest"
            kwargs = {
                'rest': ['1' for _ in range (self.n_qubits - 2)]
            }

        else:
            raise ValueError(f"Invalid input name ({name}) (obs {obs}).")
            # Note: this method could be expanded here to include more possibilities 

        cross = cross_match(key_cross, obs=obs, **kwargs)
        return cross
    

    ##################
    # Return Simulation name and description as a tuple of strings
    @abstractmethod
    def get_description(self) -> Tuple[str, str]:
        pass

    # Return model input parameters as a tuple of strings
    @abstractmethod
    def get_params(self, **kwargs) -> str:
        pass

    # Setup model-specific simulation parameters
    @abstractmethod
    def aux_setup_simulation(self, **kwargs):
        pass

    # Parse auxiliary arguments
    @abstractmethod
    def parse_aux_args(self, args):
        pass

    # Define auxiliary simulation arguments as a dictionary
    @property
    @abstractmethod
    def dict_aux_args(self):
        pass

    # Get the system Hamiltonian as a 'CrossHamiltonian' object
    @abstractmethod
    def get_hamiltonian(self) -> CrossHamiltonian:
        pass

    ###################
    ### TROTTERIZED ###
    ###################
    # System-agnostic method to run Trotter simulation
    def trotter_simulation(self, noisy=True):
        if noisy:
            backend = self.noisy_backend
            name = "Noisy"
        else:
            backend = self.exact_backend
            name = "Exact"
        
        # Array for storing Qiskit results
        trotter_results = []

        print(f"{name} Trotter start.")
        time_taken = time.time()
        results = [self._trotterize(i, backend=backend) for i in range(self._t_steps)]
        time_taken = time.time() - time_taken
        print(f"{name} Trotter end (time taken: {time_taken}).\n")

        trotter_circs, trotter_results = zip(*results)

        if self._debug:
            print("\nDebug Trotterisation:")
            print("Initial state:", self._initial_state._name) #TODO: Check
            print("Initial state + Trotter step:")
            print(trotter_circs[1])
            print("Final Circuit:")
            print(trotter_circs[-1])
            print("Circuit list size:", len(trotter_circs))
            print("Trotter Shots:", self.trot_shots)
            print("End debug Trotterisation.\n")

        save_trotter = self._save and self.has_method('trotter')

        # Save Trotter results
        if save_trotter:
            trotter_result_list = [trotter_res["result"] for trotter_res in trotter_results]
            with open(self._output_folder+f"/{name}_trotter_results{self._results_suffix}{self._data_label}.pickle", 'wb') as f:
                pickle.dump(trotter_result_list, f, pickle.HIGHEST_PROTOCOL)

        # Save Trotter measurements
        trotter_measures = []
        for i in range(self._t_steps):
            obs_res = trotter_results[i]['value']
            # obs_res = self._observable.comp_basis_prob(trotter_results[i], self.trot_shots)
            trotter_measures.append(obs_res)

        # Save observable values
        if save_trotter:
            np.savetxt(self._output_folder+f"/{name}_trotter_{self._observable.type}{self._data_label}_{self._observable.name}.out", trotter_measures)

        if not self.trot_list:
            # Store computed Trotter circuits, and toggle flag
            self.circuit_list = trotter_circs
            self.trot_list =  True

            # Save Trotter circuits
            if save_trotter:
                with open(self._output_folder+f"/trotter_circuits{self._data_label}.pickle", 'wb') as f:
                    pickle.dump(self.circuit_list, f, pickle.HIGHEST_PROTOCOL)

    # Auxiliary function for running the internal trotterization
    def _trotterize(self, ind, backend):
        if not self.trot_list:
            trot_circ = self._initial_state.as_qk_circuit.copy()

            for _ in range(ind):
                trot_circ.append(self.trotter, trot_circ.qubits)

        else:
            trot_circ = self.circuit_list[ind].copy()
        
        circ = trot_circ.copy()

        res = self._observable.run_qc(circ, backend, shots=self.trot_shots, run_estimator=self._estimator, n=self.n_qubits, optimization_level=3)

        print("{}/{}".format(ind+1, self._t_steps))
        
        return trot_circ, res
                

    # Shorthand for running exact Trotter simulation
    def exact_trotterized_simulation(self):
        self.trotter_simulation(False)

    # Shorthand for running nosiy Trotter simulation
    def noisy_trotterized_simulation(self):
        self.trotter_simulation(True)

    ###########
    ### ISL ###
    ###########
    def isl_simulation(self, noisy=True):
        # Define the backend to use
        sim_name = "Noisy" if noisy else "Exact"
        backend = self.noisy_backend if noisy else self.exact_backend

        # Compute the Trotterized circuit list if it does not exist
        if not self.trot_list:
            self.exact_trotterized_simulation()

        # Transpile circuits to speciifc backend
        max_circuit = transpile(self.circuit_list[-1], self.noisy_backend) # num of CNOTs may not exceed largest Trotter circuit
        max_layers  = max_circuit.depth()
        n_2q_gates = max_circuit.count_ops()['cx']

        if self._debug:
            print("Trot_list computed:", self.trot_list)
            print("Circuit list size:", len(self.circuit_list))
            print(f"ISL config: max_layers={max_layers}, tolerance={self.tol}, max_2q_gates={n_2q_gates}, max_layers={max_layers}")
            print("Coupling map:", self.noisy_backend.configuration().coupling_map)

        config = ISLConfig(max_layers=max_layers,
                           sufficient_cost=self.tol,
                           max_2q_gates=n_2q_gates,
                           method='ISL')
    
        # Added for identifying typical tolerance values
        name_append = ''
        if self.tol == 0.01:
            name_append = '_1e-2'
        elif self.tol == 0.0001:
            name_append = '_1e-4'

        # Run the recompilation procedure on the ISL circuits
        isl_results = []
        isl_circuits = []
        previous_circ = self._initial_state.as_qk_circuit.copy()
        isl_circuits.append(previous_circ.copy())
        print(f"\n{sim_name} ISL start.")
        time_taken = time.time()

        ### Set the proper physical indices to the trivial layout
        physical_indices = [i for i in range(self.n_qubits)]

        # Keep track of coupling map (according to transpilation changes)
        self.coupling_map = self.noisy_backend.configuration().coupling_map
        print("Coupling map:", self.coupling_map)

        for i in range(1, self._t_steps):
            target_circ = previous_circ

            target_circ.append(self.trotter, physical_indices)

            print("target circuit:\n", target_circ)

            if noisy:
                # pre-transpile the circuit for optimised gate count
                target_circ = transpile(
                    target_circ,
                    backend=self.noisy_backend,
                    optimization_level=3,
                    initial_layout=physical_indices,
                )

            recompiler_kwargs = {'isl_config': config}
            
            if not noisy:
                recompiler_kwargs['backend'] = self.exact_backend
                recompiler_kwargs['coupling_map'] = self.coupling_map
            else:
                recompiler_kwargs['backend'] = self.noisy_backend

                target_circ_layout = target_circ._layout

                # Debug: print transpiled qubits
                transpiled_qubits = get_layout_qubits(target_circ_layout, self.n_qubits)

                if self._debug:
                    print("Transpiled qubits:", transpiled_qubits)
                    print("Transpiled target circuit:")
                    print(target_circ, "\n")
                
                target_circ = truncate_to_active_qubits(target_circ, target_circ_layout)
                
                if self._debug:
                    print("Transpiled and truncated target circuit:")
                    print(target_circ, "\n")
                    print("Virtual to physical:", transpiled_qubits)

                self.coupling_map = update_coupling_map(self.coupling_map, transpiled_qubits)
                recompiler_kwargs['coupling_map'] = self.coupling_map

                if self._debug:
                    print("coupling map:", self.coupling_map)
    
            recompiler = ISLRecompiler(target_circ, execute_kwargs={'shots':self.isl_shots}, **recompiler_kwargs)
            result = recompiler.recompile()
            isl_results.append(result)
            result_circuit = result['circuit']
            previous_circ = result_circuit.copy()
            isl_circuits.append(result_circuit)

            print("Recompilation step {}/{}\t{:.4f}\t{:.2f} min".format(i, self._t_steps - 1, result['overlap'], result['time_taken']/60))

        time_taken = time.time() - time_taken
        print(f"{sim_name} ISL end (time taken: {time_taken}).\n")

        if self._debug:
            print("\nDebug ISL:")
            for i in range(self._t_steps):
                print("{} (Step {}/{})".format(isl_results[i], i, self._t_steps))
                print(isl_circuits[i])

        # Store the results from ISL into the specified output folder
        if self._save:
            with open(self._output_folder+f"/{sim_name}_isl{name_append}_results{self._data_label}.pickle", 'wb') as f:
                pickle.dump(isl_results, f, pickle.HIGHEST_PROTOCOL)

        # Run the simulations on the recompiled circuits
        isl_qiskit_results = []
        isl_measures = []
        print(f"{sim_name} ISL simulation start.")
        for i in range(self._t_steps):
            circ = isl_circuits[i].copy()
            init_layout = None # Because we are using trivial layout

            results = self._observable.run_qc(circ, backend, shots=self.isl_shots, run_estimator=self._estimator, optimization_level=3, n=self.n_qubits, init_layout=init_layout)

            result = results["result"]
            isl_qiskit_results.append(result)

            obs_res = results["value"]
            isl_measures.append(obs_res)
            print("Simulation step {}/{}".format(i+1, self._t_steps))

        print(f"{sim_name} ISL simulation end.\n")

        if self._debug:
            print("\nDebug ISL measurements:")
            for i, obs_res in enumerate(isl_measures):
                print(f"Observable: {obs_res}, (Step {i}/{self._t_steps})")
            print("End debug ISL.\n")

        # Store the resulting circuits
        if self._save:
            with open(self._output_folder+f"/{sim_name}_isl{name_append}_qiskit_results{self._results_suffix}{self._data_label}.pickle", 'wb') as f:
                pickle.dump(isl_qiskit_results, f, pickle.HIGHEST_PROTOCOL)

            # Store the resulting measurements
            np.savetxt(self._output_folder+f"/{sim_name}_isl{name_append}_{self._observable.type}{self._data_label}_{self._observable.name}.out", isl_measures)

    # Shorthand for running noisy ISL
    def noisy_isl_simulation(self):
        self.isl_simulation(True)
    
    # Shorthand for running noiseless ISL
    def exact_isl_simulation(self):
        self.isl_simulation(False) 

    ###########
    ### ZNE ###
    ###########
    # Run the mitigated results
    def zne_simulation(self):
        zne_measures = []
        exp_factory = zne.inference.ExpFactory(scale_factors=[1, 2, 3], asymptote=1/(self.n_qubits**2))
        scaling_noise = zne.scaling.fold_global

        # Define the mitigated executor
        self.mitigated_executor = zne.mitigate_executor(
            self.noisy_executor, scale_noise=scaling_noise, 
            factory=exp_factory, num_to_average=self.zne_avg
            )

        # Compute trotter list if not yet available
        if not self.trot_list:
            self.exact_trotterized_simulation()

        # Perform ZNE
        print(f"ZNE start. ({self.zne_shots} shots, {self.zne_avg} avg)")
        physical_indices = [i for i in range(self.n_qubits)]
        time_taken = time.time()
        for i, circuit in enumerate(self.circuit_list):
            print("Step {}/{}".format(i, self._t_steps))

            circ_tr = transpile(
                circuit,
                backend=self.noisy_backend,
                basis_gates=self.noise_model.basis_gates + ["save_density_matrix"] if not self._estimator else [],
                optimization_level=3, # Optimising circuit beforehand before ZNE procedure
                initial_layout = physical_indices,
            )

            circ_layout = circ_tr._layout       # Keep track of the layout for the transpilation step

            circ_tr, _ = truncate_to_active_qubits(circ_tr, circ_layout)

            zne_measures.append(self.mitigated_executor(circ_tr))

        time_taken = time.time() - time_taken
        print(f"ZNE end (time taken: {time_taken}).\n")

        if self._debug:
            print("\nDebug ZNE:")
            for i, prob in enumerate(zne_measures): 
                print("Probability:", prob)
            print("End debug ZNE.\n")

        # Save ZNE results
        saving_path = self._output_folder+f"/zne_{self._observable.type}{self._data_label}_{self._observable.name}.out"
        if self._save:
            print("Saving ZNE results to:", saving_path)
            np.savetxt(saving_path, zne_measures)
        else:
            print("Would save ZNE results to:", saving_path)


    # Define the auxiliary ZNE execute function - run a circuit on the noisy backend and compute observable
    def noisy_executor(self, circuit):
        # Copy the circuit, save the necessary information and transpile
        circuit = circuit.copy()

        exec_circ = circuit 

        # Run the circuit and obtain the observable
        results = self._observable.run_qc(exec_circ, self.noisy_backend, shots=self.zne_shots, run_estimator=self._estimator, n=self.n_qubits)
        val = results["value"]
        return val


    ##################
    ### Executions ###
    ##################
    # Procedure to run all simulations provided parameters are set  
    def run_simulations(self, **kwargs):
        if not self._set_up:
            print("Performing setup procedure.\n")
            self.setup_simulation(**kwargs)

        simulation_methods = {
            'exact': self.exact_simulation,
            'exact_trotter': self.exact_trotterized_simulation,
            'noisy_trotter': self.noisy_trotterized_simulation,
            'exact_isl': self.exact_isl_simulation,
            'noisy_isl': self.noisy_isl_simulation,
            'zne': self.zne_simulation,
        }
        
        print("\nSimulations start.")
        time_taken = time.time()
        for simulation in self._simulations:
            if simulation in simulation_methods:
                simulation_methods[simulation]()
            else:
                print(f"Error - simulation method {simulation} not recognised.")

        time_taken = time.time() - time_taken
        print(f"Simulations complete (total time taken: {time_taken}).\n")
        

    #######################
    ### Misc. method(s) ###
    #######################

    # Define the number of qubits
    @property
    @abstractmethod
    def n_qubits(self) -> int:
        pass

    # Define the simulation name
    @property
    @abstractmethod
    def simulation_name(self) -> str:
        pass


########################################
### ****** Simulation Models ******  ###
########################################
#######################
### Tavis-Cummings  ###
#######################
class TC_Simulator(Spin_Boson_Simulator):
    """
    Simulator for Tavis Cummings Models, for N atoms, parameterizable system, and user-specified encoding
    """
    def __init__(self):
        self._resonance = True     # Flag to check whether system is in resonance
        super().__init__()


    def get_description(self) -> Tuple[str, str]:
        name = "Tavis-Cummings Simulator"
        desc = "Simulate the Tavis-Cummings model using Qiskit and QuTiP."
        return (name, desc)


    def get_params(self,) -> Tuple[str, str]:
        mainparams = f"{self._n_atoms}_atoms"
        subparams = f"{self._frequencies}_omega_{self._g}_g"

        return (mainparams, subparams)


    def aux_setup_simulation(self, **kwargs):
        self._n_atoms = kwargs.get("n_atoms", 1)
        self._omega = kwargs.get("omega", 1)
        self._g = kwargs.get("g", 1)
        self._aomega = kwargs.get("aomega", -1)

        # Check if atomic frequency is not provided - set to cavity frequency (resonance)
        if self._aomega == -1:
            self._aomega = self._omega
            self._resonance = True
            self._frequencies = self._omega
        else:
            self._frequencies = (self._omega, self._aomega)

        # Computer other variables
        self._n_photon_qubits = 1
        self._n_qubits = self._n_photon_qubits + self._n_atoms
        self._cutoff = 2**self._n_photon_qubits


    def parse_aux_args(self, args):
        if args.aomega == -1:
            aomega = args.omega
        else:
            aomega = args.aomega
        
        aux_args = {
            'n_atoms': args.n_atoms,
            'omega': args.omega,
            'aomega': aomega,
            'g': args.g
        }
        return aux_args

    @property
    def dict_aux_args(self):
        aux_dict = {
            'n_atoms': self._n_atoms,
            'omega': self._omega,
            'aomega': self._aomega,
            'g': self._g
        }
        return aux_dict

    def get_hamiltonian(self):
        return Tavis_Cummings_Hamiltonian(self._n_atoms, self._frequencies, self._g, self._cutoff)

    @property
    def n_qubits(self):
        return self._n_qubits
    
    @property
    def simulation_name(self):
        if self._n_atoms > 1:
            return "Tavis-Cummings"
        else:
            return "Jaynes-Cummings"

#############################
### Heisenberg Spin Chain ###
#############################
class HS_Simulator(Spin_Boson_Simulator):
    """
    Simulator for Heisenberg Spin Chain Models, for N sites, parameterizable system, and user-specified encoding
    """

    def get_description(self) -> Tuple[str, str]:
        name = "Heisenberg Spin Simulator"
        desc = "Simulate the Heisenberg model using Qiskit and QuTiP."
        return (name, desc)
    
    def get_params(self,) -> Tuple[str, str]:
        mainparams = f"{self._n_sites}_sites"
        subparams = f"{self._h}_h_{self._J}_J"

        return (mainparams, subparams)
    
    def aux_setup_simulation(self, **kwargs):
        self._n_sites = kwargs.get("n_sites", 1)
        self._h = kwargs.get("h", [1, 1, 1])
        self._J = kwargs.get("J", [1, 1, 1])
        self._n_qubits = self._n_sites


    def parse_aux_args(self, args):
        aux_args = {
            'n_sites': args.n_sites,
            'h': args.h,
            'J': args.J
        }
        return aux_args

    @property
    def dict_aux_args(self):
        aux_dict = {
            'n_sites': self._n_sites,
            'h': self._h,
            'J': self._J
        }
        return aux_dict
    
    def get_hamiltonian(self):
        return Heisenberg_Hamiltonian(self._n_sites, self._h, self._J)
    
    @property
    def n_qubits(self):
        return self._n_qubits

    @property
    def simulation_name(self):
        name_suffix = self.H.name.rsplit("_", 1)[-1]
        return "Heisenberg Spin " + name_suffix

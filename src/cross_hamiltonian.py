# Classes for representing Hamiltonian systems

from qutip import (sigmax, sigmay, sigmaz, sigmap, 
                   sigmam, destroy, create, num, qeye, 
                   tensor, Qobj)
from math import ceil
from qiskit.quantum_info import Operator, SparsePauliOp 
import numpy as np
from typing import Union
from abc import abstractmethod

# Handling importing through different paths
from utils.cross_conversions import qobj_to_sparsepauliop
from cross_operator import InputObservable


############################
### Abstract Hamiltonian ###
############################
class CrossHamiltonian(InputObservable):
    def __init__(self, H_list: list, H_name: str):
        """
        H_list: list of all terms in the Hamiltonian,
        where each term is a list of tuples,
        where the 1st element of the tuple is a list of tensor product factors
        and the 2nd elements is the coefficient of the term
        """

        self._terms = H_list
        self._name = H_name
        data = self.map_to_qubits().to_matrix()
        mirror_data = self.full_hamiltonian().full()
        super().__init__(H_name, data, mirror_data, "energy")

    @property
    def terms(self):
        return self._terms
    
    def num_dof(self):
        """ 
        Return the number of degrees of freedom (= factors in each term) 
        """
        return len(self.terms[0][0]) 

    def full_hamiltonian(self) -> Qobj:
        """ 
        Returns the full Hamiltonian based on the list 
        """
        ham = Qobj()

        for i in range(self.num_terms):
            factors = self.terms[i][0]
            coeff = self.terms[i][1]
            ham += coeff * tensor(factors)
        return ham
    
    @property
    def num_terms(self):
        return len(self.terms)

    # Define an abstract method for flagging bosonic terms in Hamiltonian to encode
    @abstractmethod
    def bosonic_condition(self):
        pass
    

    # Define an abstract method for the encoding of bosonic terms in Hamiltonian
    @abstractmethod
    def bosonic_encoding(self, term, encoding: str):
        pass


    def map_to_qubits(self, encoding: str = 'standard') -> SparsePauliOp:
        """
        Map Hamiltonian onto qubits. Qubit order such that first op. in the factors list
        is mapped to the first qubits according to qiskit's qubit ordering.
        Args:
            encoding (str): Encoding for bosonic operators. 'standard' or 'graycode'.
        Returns:
            SparsePauliOp
        """
        for i, term in enumerate(self.terms):
            factors, coeff = term   # Factors of Hamiltonian term, along with coefficient
            for j, fact in enumerate(factors):
                if self.bosonic_condition(j):
                    op_q = self.bosonic_encoding(fact, encoding)
                else:
                    # dummy conversion, ok for Pauli operators at least
                    op_q = SparsePauliOp.from_operator(Operator(fact.data.toarray()))
                if j == 0:
                    id_str = ''.join(['I' for _ in range(op_q.num_qubits)])     # Constructs identity as string if j == 0 (i.e. index 0, corresponding to the bosonic field)
                    term_q = op_q.compose(SparsePauliOp(id_str, coeff))         # Composes Identity, along with op_q
                else:
                    term_q = term_q.expand(op_q)    # i.e. term_q = op_q (x) term_q  (tensor product)
            if i == 0:
                hamiltonian_q = term_q              # Defines Hamiltonian as term obtained from tensor products
            else:
                hamiltonian_q = SparsePauliOp.sum([hamiltonian_q, term_q])  # Otherwise, sums the new Hamiltonian with previous list
            
        return hamiltonian_q.simplify()
    

    def to_dict(self):
        """
        Convert the CrossHamiltonian instance to a dictionary.
        """
        return {
            '_type': "CrossHamiltonian",
            'H_name': self._name,
            'H_list': [[(self._tensor_to_dict(factor)) for factor in term[0]] + [term[1]] for term in self.terms]
        }


    @staticmethod
    def _tensor_to_dict(tensor):
        """
        Helper method to convert a tensor to a dictionary.
        """
        return {
            'data': tensor,
        }
    

    @classmethod
    def from_dict(cls, dict_obj):
        """
        Create a CrossHamiltonian instance from a dictionary.
        """
        print(type(dict_obj))
        print(dict_obj.keys())
        H_list = [[cls._dict_to_tensor(factor) for factor in term[:-1]]  + [term[-1]] for term in dict_obj["H_list"]]
        H_name = dict_obj["H_name"]
        return cls(H_list, H_name)
    

    @staticmethod
    def _dict_to_tensor(dict_obj):
        """
        Helper method to convert a dictionary back to a tensor product factor.
        You may need to adjust this method based on the actual structure of your tensor factors.
        """
        qobj = Qobj(
            data=np.array(dict_obj["data"]),
            dims=dict_obj["dims"],
            shape=dict_obj["shape"],
            type=dict_obj["type"]
        )
        return qobj

###############################
### Spin-Boson Hamiltonians ###
###############################
# Define abstract class for spin-boson systems
class SpinBosonHL(CrossHamiltonian):
    def __init__(self, H_list: list, H_name: str, photon_indices: list[int], cutoff: int):
        """ 
        H_list must be a list of all terms in the Hamiltonian,
        where each term is a list of tuples,
        where the 1st element of the tuple is a list of tensor product factors
        and the 2nd element is the coefficient of the term.
        """

        self._terms = H_list
        self._photon_ind = photon_indices
        self._cutoff = cutoff   # Define the cutoff for the dimension of the bosonic field
        super().__init__(H_list, H_name)

    # Define indices with an initial excitation
    @property
    def photon_ind(self):
        return self._photon_ind
    
    @property
    def cutoff(self):
        return self._cutoff
    
    def qubits_per_photon(self):
        return ceil(np.log2(self.cutoff))

    def num_photons(self):
        return len(self.photon_ind)

    def bosonic_condition(self, j: int):
        return j in self.photon_ind

    def bosonic_encoding(self, fact, encoding):
        return qobj_to_sparsepauliop(fact, self.qubits_per_photon(), encoding=encoding)


# ****** Define commonly used Hamiltonians ****** 
######################
### TC Hamiltonian ###
######################
class Tavis_Cummings_Hamiltonian(SpinBosonHL):
    def __init__(self, n_atoms: int, omega: Union[float, tuple], g: float = 1., cutoff: int = 2, photon_ind: list[int] = [0], **kwargs):
        """
        Tavis Cummings Hamiltonian.

        Args:
            n_atoms (int): Number of atoms in the system.
            omega (Union[float, tuple]): Photonic mode frequency or a tuple of photonic mode frequency and atomic frequency.
            g (int): Interaction strength.
            cutoff (int): Cutoff for photon dimension.
            photon_ind (list[int]): List of indices of elements in the system which are initially excited.
        """
        # Instantiate the frequencies according to the input type
        if isinstance(omega, tuple):
            omega_cav, omega_atom = omega
        else:
            omega_cav = omega
            omega_atom = omega

        # Here we assume that the atoms are in resonance. TODO: consider general case. TODO: Scale by 1/2 here?
        # Cavity Hamiltonian
        h_ph = ([num(cutoff)] + [qeye(2) for _ in range(n_atoms)], omega_cav)

        # TLS Hamiltonian(s)
        h_tls = lambda i : ([qeye(cutoff)] + [sigmaz() if i==j else qeye(2) for j in range(n_atoms)], omega_atom*0.5)
        
        # Interaction Hamiltonian(s)
        h_int1 = lambda i: ([create(cutoff)] + [sigmam() if i==j else qeye(2) for j in range(n_atoms)], g)
        h_int2 = lambda i : ([destroy(cutoff)] + [sigmap() if i==j else qeye(2) for j in range(n_atoms)], g)

        # Define CrossHamiltonian parameters
        h_list = [h_ph] + [h_tls(i) for i in range(n_atoms)] + [h_int1(i) for i in range(n_atoms)] + \
                [h_int2(i) for i in range(n_atoms)]

        name = "TC_Ham" if n_atoms > 1 else "JC_Ham"


        # Define initial params, for dictionary conversion
        self.init_params = {
            'n_atoms': n_atoms,
            'omega': omega,
            'g': g,
            'cutoff': cutoff,
            'photon_ind': photon_ind,
        }

        super().__init__(h_list, name, photon_ind, cutoff)


    def to_dict(self):
        dict_obj = self.init_params.copy()
        dict_obj['_type'] = "TC_Hamiltonian"     # TOD: add to json_conversions?
        return dict_obj
    

    @staticmethod
    def from_dict(d):
        d.pop('_type')
        return Tavis_Cummings_Hamiltonian(**d)


######################
### JC Hamiltonian ###
######################
class Jaynes_Cummings_Hamiltonian(Tavis_Cummings_Hamiltonian):
    def __init__(self, delta: int = 0, g: int = 1, cutoff: int=2):
        """Jaynes-Cummings Hamiltonian.
            Args:
                Delta: Detuning.
                g: Interaction strength.
                cutoff: Cutoff for photon dimension.a
        """

        n_atoms = 1
        omega = (0, delta) # Here we set the cavity frequency to 0 as it is not relevant for the JC model
        super().__init__(n_atoms, omega=omega, g=g, cutoff=cutoff)


#########################
### Spin Hamiltonians ###
#########################
# Hamiltonians with no bosonic field (i.e. no encoding required)
class SpinHamiltonian(CrossHamiltonian):
    def __init__(self, H_list: list, H_name: str):
        """ 
        n_atoms: Number of atoms in the system.
        """
        super().__init__(H_list, H_name)

    # No bosonic condition or encoding
    def bosonic_condition(self, _):
        return False

    def bosonic_encoding(self, _):
        pass


#########################
### Spin Hamiltonians ###
#########################
# Class for (Heisenberg Spin (1D) chain Hamiltonians
class Heisenberg_Hamiltonian(SpinHamiltonian):
    def __init__(self, n_sites: int = None, h: list[float] = None, J: list[float] = None):
        """
        H_list must be a list of all terms in the Hamiltonian,
        where each term is a list of tuples,
        where the 1st element of the tuple is a list of tensor product factors
        and the 2nd element is the coefficient of the term.
        """
        
        self._n_sites = n_sites
        
        # Initialise Hamiltonian list
        h_list = []

        ops = [sigmax(), sigmay(), sigmaz()]

        # **** Define the Hamiltonian terms ****
        for k in range(3):
            # Magnetic-field terms
            if h[k] != 0:
                h_field = [lambda i : ([ops[k] if i==j else qeye(2) for j in range(n_sites)], -1/2 * h[k])]
                h_list += [H(i) for i in range(n_sites) for H in h_field]

            # Coupling terms
            if J[k] != 0:
                h_int = [lambda i : ([ops[k] if self.boundary(i, j) else qeye(2) for j in range(n_sites)], -1/2 * J[k])]
                h_list += [H(i) for i in range(n_sites) for H in h_int]

        # Define the name of the Hamiltonian
        name = "Heis_Ham_"
        Jx, Jy, Jz = J

        if Jx == Jz == 0 and Jz != 0:
            name += "Z"     # Ising spin chain

        elif Jz == 0 and Jx == Jy != 0:
            name += "XX"    # XX spin chain (equivalent to a free lattice fermion)

        elif Jx == Jy == Jz:
            name += "XXX"   # Isotropic case (XXX spin chain)

        elif Jx == Jy or Jy == Jz or Jx == Jz:
            name += "XXZ"   # Anisotropic case (XXZ spin chain)

        else:
            name += "XYZ" # Completely anisotropic case (XYZ spin chain)


        self.init_params = {
            'n_sites': n_sites,
            'h': h,
            'J': J,
        }

        super().__init__(h_list, name)

    # Define the boundary condition of the model
    def boundary(self, i, j, periodic: bool = True):
        if periodic:
            return (i == j) or ((i + 1) % self._n_sites == j) # Periodic boundary conditions are assumed
        else:
            print("ERROR - alternative boundary conditions not implemented.")
            return (i == j)

    def to_dict(self):
        dict_obj = self.init_params.copy()
        dict_obj['_type'] = "HS_Hamiltonian"
        return dict_obj
    
    @staticmethod
    def from_dict(d):
        d.pop('_type')
        return Heisenberg_Hamiltonian(**d)
    
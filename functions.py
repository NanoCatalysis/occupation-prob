# Third party imports
import numpy as np
import ase
import matplotlib.pyplot as plt

def read_properties(isomers):
    n_isomers = len(isomers) # Numbers of isomers
    n_atoms = isomers[0].positions.shape[0] # Number of atoms in each isomer
    
    # Number of vibrational modes
    n_freq = (3*n_atoms - 6) if n_atoms > 2 else (3*n_atoms - 5)
    
    # Properties arrays
    energies = np.zeros([n_isomers,])
    symm_order = np.zeros([n_isomers,])
    spin_m = np.zeros([n_isomers,])
    frequencies = np.zeros([n_isomers, n_freq])
    moments = np.zeros([len(isomers), 3])
    
    # Reads the values from the input file
    for n, atoms in enumerate(isomers):
        energies[n] = atoms.info['energy']
        symm_order[n] = atoms.info['symm_order']
        spin_m[n] = atoms.info['multiplicity']
        frequencies[n] = atoms.info['frequencies'].flatten(order='F')
        moments[n] = atoms.get_moments_of_inertia()
        
    # Converts the energy values to energy differences respect to the minimum
    energies -= energies.min()
        
    return energies, symm_order, spin_m, frequencies, moments

def calc_prob(energies, symm_order, spin_m, freq, moments, max_temp, n_temp):
    # Number of isomers
    n_isomers = energies.size
    
    # Temperature array
    temp = np.linspace(0.01, max_temp, num=n_temp)
    temp /= 1.5788732e5
    
    # Partition functions for each isomer
    z_k = np.zeros([n_isomers, n_temp])

    # Multiplies the principal moments of inertia
    moments_prod = np.prod(moments, axis=1)
    
    # Quantum superposition
    for j in range(n_temp):
        # Vibrational partition function
        z_vib_a = np.exp(-np.pi * freq / temp[j])
        z_vib_a /= 1. - np.exp(-2.*np.pi * freq / temp[j])
        z_vib = np.prod(z_vib_a, axis=1)
        
        # Rotational partition function
        z_rot = np.sqrt(8.*np.pi * temp[j]**3 * moments_prod)
        
        for i in range(n_isomers):
            z_k[i, j] = spin_m[i] * np.exp(-energies[i]/temp[j]) / symm_order[i]
            z_k[i, j] *= z_vib[i] * z_rot[i]
    
    temp *= 1.5788732e5
    
    # Total partition function
    z_tot = np.sum(z_k, axis=0)
    
    # Occupation probabilities for each isomers
    prob = z_k / z_tot
    
    return temp, prob

def plot_prob(temp, prob, outfile, size):
    # Plot
    plt.figure(figsize=size) 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['font.size'] = '12'
    
    plt.xlabel(r'$T$ [K]')
    plt.ylabel(r'$P(T)$')
    plt.xticks((0, 200, 400, 600, 800, 1000, 1200))
    plt.xlim((0, 1200))
    plt.ylim((0, 1))
    
    for i in range(prob.shape[0]):
        if i == 0:
            isoLabel = 'GM'
        else:
            isoLabel = 'ISO' + str(i)
        
        plt.plot(temp, prob[i], label=isoLabel, lw=2)
        
    plt.tight_layout()
    plt.savefig(outfile)
    
    return None
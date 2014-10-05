#! /usr/bin/env python3
"""
monte-carlo-liquid.py

This script runs the Monte Carlo uncertainty analysis for
a liquid fuel in the University of Connecticut RCM. This script is
associated with the work "On the Uncertainty of Temperature Estimation
in a Rapid Compression Machine" by Bryan W. Weber, Chih-Jen Sung, and
Michael Renfro, submitted to Combustion and Flame. This script is
licensed according to the LICENSE file available in the repository
associated in the paper.

Please email bryan@engr.uconn.edu with any questions.
"""

# System library imports
import sys
from multiprocessing import Pool
from itertools import repeat as rp

if sys.version_info[:2] < (3, 3):
    print('This script requires Python 3.3 or higher.')
    sys.exit(1)

try:
    from scipy.special import lambertw
    from scipy.stats import norm as norm_dist
    from scipy.stats import triang
    from scipy.stats import uniform
except ImportError:
    print('SciPy must be installed')
    sys.exit(1)

try:
    import cantera as ct
except ImportError:
    print('Cantera must be installed')
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print('NumPy must be installed')
    sys.exit(1)

def run_case(dummy, fuel, P_0, T_0, P_C, mix):
    # Set the Cantera Solution with the thermo data from the xml file.
    # Set the unit basis for the Solution to molar basis.
    gas = ct.Solution('therm-data.xml')
    gas.basis = 'molar'

    # Create the normal distributions for the initial temperature,
    # initial pressure, and compressed pressure. Convert the initial
    # temperature to °C to match the spec. Use the appropriate
    # distribution for the desired analysis (normal, uniform,
    # triangular).
    sigma_T0 = max(2.2, (T_0 - 273)*0.0075)/2
    T0_dist = norm_dist(loc=T_0, scale=sigma_T0)
    # T0_dist = triang(loc=T_0-sigma_T0, scale=2*sigma_T0, c=0.5)
    # T0_dist = uniform(loc=T_0-sigma_T0, scale=2*sigma_T0)

    sigma_P0 = 346.6/2
    P0_dist = norm_dist(loc=P_0, scale=sigma_P0)

    sigma_PC = 5000/2
    PC_dist = norm_dist(loc=P_C, scale=sigma_PC)

    sigma_pressure = 346.6/2
    # For the experiments studied here, if there was no argon added,
    # the nitrogen was added first. Otherwise, the order was o2, n2,
    # ar, fuel. Compute the pressures of each component as they are
    # filled into the mixing tank.
    if mix['ar'] > 0:
        nom_o2_pres = mix['o2']*ct.one_atm/760
        nom_n2_pres = (mix['n2'] - mix['o2'])*ct.one_atm/760
        nom_ar_pres = (mix['ar'] - mix['n2'])*ct.one_atm/760
        nom_fuel_pres = (mix['fuel'] - mix['ar'])*ct.one_atm/760
        o2_dist = norm_dist(loc=nom_o2_pres, scale=sigma_pressure)
        o2_pres_rand = o2_dist.ppf(np.random.random_sample())
        n2_pressure = o2_pres_rand + nom_n2_pres
        n2_dist = norm_dist(loc=n2_pressure, scale=sigma_pressure)
        n2_pres_rand = n2_dist.ppf(np.random.random_sample())
        ar_pressure = n2_pres_rand + nom_ar_pres
        ar_dist = norm_dist(loc=ar_pressure, scale=sigma_pressure)
        ar_pres_rand = ar_dist.ppf(np.random.random_sample())
        fuel_pressure = ar_pres_rand + nom_fuel_pres
        fuel_dist = norm_dist(loc=fuel_pressure, scale=sigma_pressure)
        fuel_pres_rand = fuel_dist.ppf(np.random.random_sample())
        o2_par_pres = o2_pres_rand
        n2_par_pres = n2_pres_rand - o2_pres_rand
        ar_par_pres = ar_pres_rand - n2_pres_rand
        fuel_par_pres = fuel_pres_rand - ar_pres_rand
    else:
        nom_n2_pres = mix['n2']*ct.one_atm/760
        nom_o2_pres = (mix['o2'] - mix['n2'])*ct.one_atm/760
        nom_fuel_pres = (mix['fuel'] - mix['o2'])*ct.one_atm/760
        nom_ar_pres = 0
        n2_dist = norm_dist(loc=nom_n2_pres, scale=sigma_pressure)
        n2_pres_rand = n2_dist.ppf(np.random.random_sample())
        o2_pressure = n2_pres_rand + nom_o2_pres
        o2_dist = norm_dist(loc=o2_pressure, scale=sigma_pressure)
        o2_pres_rand = o2_dist.ppf(np.random.random_sample())
        fuel_pressure = o2_pres_rand + nom_fuel_pres
        fuel_dist = norm_dist(loc=fuel_pressure, scale=sigma_pressure)
        fuel_pres_rand = fuel_dist.ppf(np.random.random_sample())
        n2_par_pres = n2_pres_rand
        o2_par_pres = o2_pres_rand - n2_pres_rand
        fuel_par_pres = fuel_pres_rand - o2_pres_rand
        ar_par_pres = 0

    # Compute the mole fractions of each component and set the state of
    # the Cantera solution.
    total_pres = sum([o2_par_pres, n2_par_pres, ar_par_pres, fuel_par_pres])
    mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
        fuel_name=fuel, fuel_mole=fuel_par_pres/total_pres,
        o2=o2_par_pres/total_pres, n2=n2_par_pres/total_pres,
        ar=ar_par_pres/total_pres)
    gas.TPX = None, None, mole_fractions

    # Initialize the array of temperatures over which the C_p should be
    # fit. The range is [first input, second input) with increment set
    # by the third input. Loop through the temperatures and compute the
    # non-dimensional specific heats.
    temperatures = np.arange(300, 1105, 5)
    gas_cp = np.zeros(len(temperatures))
    for j, temp in enumerate(temperatures):
        gas.TP = temp, None
        gas_cp[j] = gas.cp/ct.gas_constant

    # Compute the linear fit to the specific heat.
    (gas_b, gas_a) = np.polyfit(temperatures, gas_cp, 1)

    # Sample the values for the initial temperature, initial pressure, and
    # compressed pressure.
    T0_rand = T0_dist.ppf(np.random.random_sample())
    P0_rand = P0_dist.ppf(np.random.random_sample())
    PC_rand = PC_dist.ppf(np.random.random_sample())

    # Compute the compressed temperature and return it.
    lam_rand = gas_b/gas_a * np.exp(gas_b*T0_rand/gas_a) * T0_rand * (PC_rand/P0_rand)**(1/gas_a)
    TC_rand = np.real(gas_a * lambertw(lam_rand)/gas_b)
    return TC_rand

if __name__ == "__main__":
    # n_runs is the number iterations to run per case
    n_runs = 1000000

    # Set the parameters to be studied so that we can use a loop
    P0s = [0.3613E5, 0.5612E5, 1.2547E5, 1.8601E5, 1.6951E5, 1.7138E5, 0.3763E5, 0.5619E5,]
    T0s = [413, 413, 413, 398, 398, 358, 413, 413,]
    PCs = [10.0918E5, 10.1670E5, 40.6010E5, 40.5571E5, 40.7850E5, 40.5031E5, 10.1301E5, 10.0784E5,]
    cases = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # Set the string of the fuel.
    pass_fuel = 'c3h6'

    # Set the mixtures to study
    mix1 = {'o2':257, 'n2':739, 'ar':1223, 'fuel':1280,}
    mix2 = {'ar':0, 'n2':1413, 'o2':1473, 'fuel':1500,}
    mix3 = {'ar':0, 'n2':1607, 'o2':1675, 'fuel':1706,}
    mix4 = {'ar':0, 'n2':1536, 'o2':1752, 'fuel':1800,}
    mix5 = {'ar':0, 'n2':1365, 'o2':1557, 'fuel':1600,}
    mix6 = {'o2':216, 'n2':984, 'ar':1752, 'fuel':1800,}

    for i, case in enumerate(cases):
        # Each case is associated with a particular mixture in the
        # paper. Set which mixture to use here.
        if case == 'a' or case == 'b':
            pass_mix = mix1
        elif case == 'c':
            pass_mix = mix2
        elif case == 'd':
            pass_mix = mix3
        elif case == 'e':
            pass_mix = mix4
        elif case == 'f':
            pass_mix = mix5
        elif case == 'g' or case == 'h':
            pass_mix = mix6

        # Set all the other initial variables and create a zip to send
        # to the run_case function.
        pass_P_0 = P0s[i]
        pass_T_0 = T0s[i]
        pass_P_C = PCs[i]
        send = zip(range(n_runs), rp(pass_fuel), rp(pass_P_0), rp(pass_T_0), rp(pass_P_C), rp(pass_mix))

        # Set up a pool of processors to run in parallel
        with Pool(processes=5) as pool:

        # Run the analysis and get the result into a NumPy array.
            result = np.array(pool.starmap(run_case, send))

        # Print the mean and twice the standard deviation to a file.
        with open('results.txt', 'a') as output:
            print(case, result.mean(), result.std()*2, file=output)

        # Create and save the histogram data file. The format is:
        # Mean temperature, standard deviation
        # Bin edges, height
        # Note the bin edges are one element longer than the histogram
        # so we append a zero at the end of the histogram.
        hist, bin_edges = np.histogram(result, bins=100, density=True)
        np.savetxt('histogram/histogram-gas-'+case+'.dat',
                   np.vstack((np.insert(bin_edges, 0, result.mean()), np.insert(np.append(hist, 0), 0, result.std()))).T
                  )

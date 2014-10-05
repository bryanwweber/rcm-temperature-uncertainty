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

def run_case(dummy, fuel, P_0, T_0, P_C, mfuel, T_a, mix):
    # Set the Cantera Solution with the thermo data from the xml file.
    # Get the molecular weight of the fuel and set the unit basis for
    # the Solution to molar basis.
    gas = ct.Solution('therm-data.xml')
    fuel_mw = gas.molecular_weights[gas.species_index(fuel)]
    gas.basis = 'molar'

    # Set the ambient temperature and distribution. Convert the ambient
    # temperature to °C to match the spec but use absolute temperature
    # for the distribution.
    sigma_Ta = max(2.2, (T_a - 273.15)*0.0075)/2
    Ta_dist = norm_dist(loc=T_a, scale=sigma_Ta)
    # Ta_dist = uniform(loc=T_a-sigma_Ta, scale=sigma_Ta*2)
    # Ta_dist = triang(loc=T_a-sigma_Ta, scale=sigma_Ta*2, c=0.5)

    # Set the tank volume and distribution.
    nom_tank_volume = 0.01660
    sigma_volume = 0.00001
    vol_dist = norm_dist(loc=nom_tank_volume, scale=sigma_volume)

    # Create the normal distributions for the initial temperature,
    # initial pressure, and compressed pressure. Convert the initial
    # temperature to °C to match the spec. Use the appropriate
    # distribution for the desired analysis (normal, uniform,
    # triangular).
    sigma_T0 = max(2.2, (T_0 - 273)*0.0075)/2
    T0_dist = norm_dist(loc=T_0, scale=sigma_T0)
    # T0_dist = uniform(loc=T_0-sigma_T0, scale=sigma_T0*2)
    # T0_dist = triang(loc=T_0-sigma_T0, scale=sigma_T0*2, c=0.5)

    sigma_P0 = 346.6/2
    P0_dist = norm_dist(loc=P_0, scale=sigma_P0)

    sigma_PC = 5000/2
    PC_dist = norm_dist(loc=P_C, scale=sigma_PC)

    # Set the nominal injected mass of the fuel. Compute the nominal
    # moles of fuel and corresponding nominal required number of moles
    # of the gases.
    nom_mass_fuel = mfuel
    nom_mole_fuel = nom_mass_fuel/fuel_mw
    nom_mole_o2 = nom_mole_fuel*mix[0]
    nom_mole_n2 = nom_mole_fuel*mix[1]
    nom_mole_ar = nom_mole_fuel*mix[2]

    # Create the normal distribution for the fuel mass.
    sigma_mass = 0.03/2
    fuel_mass_dist = norm_dist(loc=nom_mass_fuel, scale=sigma_mass)

    # Calculate the nominal pressure required for each gas to match the
    # desired molar proportions. Note that the gas constant from
    # Cantera is given in units of J/kmol-K, hence the factor of 1000.
    nom_o2_pres = nom_mole_o2*ct.gas_constant*T_a/(1000*nom_tank_volume)
    nom_n2_pres = nom_mole_n2*ct.gas_constant*T_a/(1000*nom_tank_volume)
    nom_ar_pres = nom_mole_ar*ct.gas_constant*T_a/(1000*nom_tank_volume)

    # Compute the pressures of each component as they are filled into
    # the mixing tank. The mean of the distribution of the pressure of
    # each subsequent gas is the sum of the sampled value of the
    # pressure of the previous gas plus the nominal value of the
    # current gas. Note that these are thus not partial pressures, but
    # the total pressure in the tank after filling each component.
    sigma_pressure = 346.6/2
    o2_dist = norm_dist(loc=nom_o2_pres, scale=sigma_pressure)
    o2_pres_rand = o2_dist.ppf(np.random.random_sample())
    n2_pressure = o2_pres_rand + nom_n2_pres
    n2_dist = norm_dist(loc=n2_pressure, scale=sigma_pressure)
    n2_pres_rand = n2_dist.ppf(np.random.random_sample())
    ar_pressure = n2_pres_rand + nom_ar_pres
    ar_dist = norm_dist(loc=ar_pressure, scale=sigma_pressure)
    ar_pres_rand = ar_dist.ppf(np.random.random_sample())

    # Sample random values of the ambient temperature, tank volume, and
    # fuel mass from their distributions.
    Ta_rand = Ta_dist.ppf(np.random.random_sample())
    tank_volume_rand = vol_dist.ppf(np.random.random_sample())
    mole_fuel_rand = fuel_mass_dist.ppf(np.random.random_sample())/fuel_mw

    # Compute the number of moles of each gaseous component based on
    # the sampling from the various distributions. Note that the gas
    # constant from Cantera is given in units of J/kmol-K, hence the
    # factor of 1000.
    mole_o2_rand = o2_pres_rand*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)
    mole_n2_rand = (n2_pres_rand - o2_pres_rand)*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)
    mole_ar_rand = (ar_pres_rand - n2_pres_rand)*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)

    # Compute the mole fractions of each component and set the state of
    # the Cantera solution.
    total_moles = sum([mole_fuel_rand, mole_o2_rand, mole_n2_rand, mole_ar_rand])
    mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
        fuel_name=fuel, fuel_mole=mole_fuel_rand/total_moles, o2=mole_o2_rand/total_moles,
        n2=mole_n2_rand/total_moles, ar=mole_ar_rand/total_moles)
    gas.TPX = None, None, mole_fractions

    # Initialize the array of temperatures over which the C_p should be fit.
    # The range is [first input, second input) with increment set by the third
    # input. Loop through the temperatures and compute the non-dimensional
    # specific heats.
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
    P0s = [1.8794E5, 4.3787E5, 3.9691E5, 4.3635E5, 1.9118E5, 4.3987E5,]
    T0s = [308]*6
    PCs = [50.0135E5, 49.8629E5, 50.0485E5, 49.6995E5, 49.8254E5, 50.0202E5,]
    mfuels = [3.43, 3.48, 3.49, 3.53, 3.53, 3.69,]
    Tas = [21.7, 21.7, 22.0, 22.1, 21.7, 20.0,]
    cases = ['a', 'b', 'c', 'd', 'e', 'f',]

    # Set the string of the fuel.
    pass_fuel = 'mch'

    # Set the mixtures to study
    mix1 = [10.5, 12.25, 71.75,]
    mix2 = [21.0, 00.00, 73.50,]
    mix3 = [07.0, 16.35, 71.15,]

    for i, case in enumerate(cases):
        # Each case is associated with a particular mixture in the
        # paper. Set which mixture to use here.
        if case == 'a' or case == 'b':
            pass_mix = mix1
        elif case == 'c' or case == 'd':
            pass_mix = mix2
        else:
            pass_mix = mix3

        # Set all the other initial variables and create a zip to send
        # to the run_case function.
        pass_P_0 = P0s[i]
        pass_T_0 = T0s[i]
        pass_P_C = PCs[i]
        pass_mfuel = mfuels[i]
        pass_T_a = Tas[i] + 273.15
        send = zip(range(n_runs), rp(pass_fuel), rp(pass_P_0), rp(pass_T_0),
                   rp(pass_P_C), rp(pass_mfuel), rp(pass_T_a), rp(pass_mix)
                  )

        # Set up a pool of processors to run in parallel.
        with Pool(processes=10) as pool:

        # Run the analysis and get the result into a NumPy array.
            result = np.array(pool.starmap(run_case, send))

        # Print the mean and twice the standard deviation to a file.
        with open('results-liquid.txt', 'a') as output:
            print(case+'tri', result.mean(), result.std()*2, file=output)

        # Create and save the histogram data file. The format is:
        # Mean temperature, standard deviation
        # Bin edges, height
        # Note the bin edges are one element longer than the histogram
        # so we append a zero at the end of the histogram.
        hist, bin_edges = np.histogram(result, bins=100, density=True)
        np.savetxt('histogram/histogram-liquid-uni'+case+'.dat',
                   np.vstack((np.insert(bin_edges, 0, result.mean()), np.insert(np.append(hist, 0), 0, result.std()))).T
                  )

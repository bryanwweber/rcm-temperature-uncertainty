#! /usr/bin/env python3
from __future__ import division, print_function
from scipy.special import lambertw
from scipy.stats import norm as norm_dist
import cantera as ct
import numpy as np
from multiprocessing import Pool
import time
import subprocess
import os
from itertools import repeat as rp

def run_case(n, fuel, P0, T0, PC, mfuel, Ta, mix):
    # Set the Cantera Solution with the thermo data from the xml file.
    # Get the molecular weight of the fuel and set the unit basis for the
    # Solution to molar basis.
    gas = ct.Solution('therm-data.xml')
    fuel_mw = gas.molecular_weights[gas.species_index(fuel)]
    gas.basis = 'molar'

    # Set the ambient temperature and tank volume
    # Ta = 21.7+273.15
    # Convert the ambient temperature to °C to match the spec.
    sigma_Ta = max(2.2, (Ta - 273.15)*0.0075)/2
    Ta_dist = norm_dist(loc=Ta, scale=sigma_Ta)
    nom_tank_volume = 0.01660
    sigma_volume = 0.00001
    vol_dist = norm_dist(loc=nom_tank_volume, scale=sigma_volume)

    # Set the initial temperature (in K), initial pressure (in Pa), and
    # compressed pressure (in Pa). Create the normal distributions for each
    # of these parameters.
    # T0 = 295.1
    # Convert the initial temperature to °C to match the spec.
    sigma_T0 = max(2.2, (T0 - 273)*0.0075)/2
    T0_dist = norm_dist(loc=T0, scale=sigma_T0)

    # P0 = 122656.579
    sigma_P0 = 346.6/2
    P0_dist = norm_dist(loc=P0, scale=sigma_P0)

    # PC = 15.1E5
    sigma_PC = 5000/2
    PC_dist = norm_dist(loc=PC, scale=sigma_PC)

    # Set the nominal injected mass of the fuel. Compute the nominal moles of
    # fuel and corresponding nominal required number of moles of the gases.
    nom_mass_fuel = mfuel
    nom_mole_fuel = nom_mass_fuel/fuel_mw
    nom_mole_o2 = nom_mole_fuel*mix[0]
    nom_mole_n2 = nom_mole_fuel*mix[1]
    nom_mole_ar = nom_mole_fuel*mix[2]

    # Create the normal distribution for the fuel mass
    sigma_mass = 0.04/2
    fuel_mass_dist = norm_dist(loc=nom_mass_fuel, scale=sigma_mass)

    # Calculate the nominal pressure required for each gas to match the desired
    # molar proportions. Note that the gas constant from Cantera is given in
    # units of J/kmol-K, hence the factor of 1000.
    nom_o2_pres = nom_mole_o2*ct.gas_constant*Ta/(1000*nom_tank_volume)
    nom_n2_pres = nom_mole_n2*ct.gas_constant*Ta/(1000*nom_tank_volume)
    nom_ar_pres = nom_mole_ar*ct.gas_constant*Ta/(1000*nom_tank_volume)

    # Compute the pressures of each component as they are filled into the
    # mixing tank. The mean of the distribution of the pressure of each
    # subsequent gas is the sum of the sampled value of the pressure of the
    # previous gas plus the nominal value of the current gas. Note that these
    # are thus not partial pressures, but the total pressure in the tank after
    # filling each component.
    sigma_pressure = 346.6/2
    o2_dist = norm_dist(loc=nom_o2_pres, scale=sigma_pressure)
    o2_pres_rand = o2_dist.ppf(np.random.random_sample())
    n2_pressure = o2_pres_rand + nom_n2_pres
    n2_dist = norm_dist(loc=n2_pressure, scale=sigma_pressure)
    n2_pres_rand = n2_dist.ppf(np.random.random_sample())
    ar_pressure = n2_pres_rand + nom_ar_pres
    ar_dist = norm_dist(loc=ar_pressure, scale=sigma_pressure)
    ar_pres_rand = ar_dist.ppf(np.random.random_sample())

    # Sample random values of the ambient temperature, tank volume, and fuel
    # mass from their distributions.
    Ta_rand = Ta_dist.ppf(np.random.random_sample())
    tank_volume_rand = vol_dist.ppf(np.random.random_sample())
    mole_fuel_rand = fuel_mass_dist.ppf(np.random.random_sample())/fuel_mw

    # Compute the number of moles of each gaseous component based on the
    # sampling from the various distributions. Note that the gas constant from
    # Cantera is given in units of J/kmol-K, hence the factor of 1000.
    mole_o2_rand = o2_pres_rand*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)
    mole_n2_rand = (n2_pres_rand - o2_pres_rand)*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)
    mole_ar_rand = (ar_pres_rand - n2_pres_rand)*tank_volume_rand*1000/(ct.gas_constant*Ta_rand)

    # Compute the mole fractions of each component and set the state of the
    # Cantera solution.
    total_moles = sum([mole_fuel_rand, mole_o2_rand, mole_n2_rand, mole_ar_rand])
    mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
        fuel_name=fuel, fuel_mole=mole_fuel_rand/total_moles, o2=mole_o2_rand/total_moles,
        n2=mole_n2_rand/total_moles, ar=mole_ar_rand/total_moles)
    gas.TPX = None, None, mole_fractions

    # Initialize the array of temperatures over which the C_p should be fit.
    # The range is [first input, second input) with increment set by the third
    # input. Loop through the temperatures and compute the non-dimensional
    # specific heats.
    temperatures = np.arange(300,1105,5)
    gas_cp = np.zeros(len(temperatures))
    for i, temp in enumerate(temperatures):
        gas.TP = temp, None
        gas_cp[i] = gas.cp/ct.gas_constant

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
    # n is the number iterations to run per case
    n = 1000000

    # Set the parameters to be studied so that we can use a loop
    P0s = [1.8794E5, 4.3787E5, 3.9691E5, 4.3635E5, 1.9118E5, 4.3987E5,]
    T0s = [308]*6
    PCs = [50.0135E5, 49.8629E5, 50.0485E5, 49.6995E5, 49.8254E5, 50.0202E5,]
    mfuels = [3.43, 3.48, 3.49, 3.53, 3.53, 3.69,]
    Tas = [21.7, 21.7, 22.0, 22.1, 21.7, 20.0,]
    cases = ['a', 'b', 'c', 'd', 'e', 'f',]

    # Set the string of the fuel. Possible values with the distributed
    # therm-data.xml are 'mch', 'nc4h9oh', 'sc4h9oh', 'tc4h9oh', 'ic4h9oh',
    # 'ic5h11oh', and 'c3h6'
    fuel = 'mch'

    # Set the mixtures to study
    mix1 = [10.5, 12.25, 71.75,]
    mix2 = [21.0, 00.00, 73.50,]
    mix3 = [07.0, 16.35, 71.15,]
    for i, case in enumerate(cases):
        start = time.time()
        if case == 'a' or case == 'b':
            mix = mix1
        elif case == 'c' or case == 'd':
            mix = mix2
        else:
            mix = mix3

        P0 = P0s[i]
        T0 = T0s[i]
        PC = PCs[i]
        mfuel = mfuels[i]
        Ta = Tas[i]
        send = zip(range(n), rp(fuel), rp(P0), rp(T0), rp(PC), rp(mfuel), rp(Ta), rp(mix))
        # Set up a pool of processors to run in parallel
        with Pool(processes=20) as pool:

        # Run the analysis and get the result into a NumPy array.
            result = np.array(pool.starmap(run_case, send))

        # Print the mean and twice the standard deviation to a file.
        with open('results.txt', 'a') as output:
            print(case, result.mean(), result.std()*2, file=output)
        print(time.time() - start)

        # Create and save the histogram data file. Compile the TeX file to make a
        # PDF figure of the histogram.
        hist, bin_edges = np.histogram(result, bins=100, density=True)
        np.savetxt('histogram/histogram-'+case+'.dat', np.vstack((np.insert(bin_edges, 0, result.mean()), np.insert(np.append(hist,0), 0, result.std()))).T)
    # os.chdir('histogram')
    # subprocess.call(['pdflatex', '-interaction=batchmode', 'histogram'])
    # print(result)
    # n, bins, patches = plt.hist(result, 100, normed=1, facecolor='green', alpha=0.75)

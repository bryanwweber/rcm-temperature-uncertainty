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

def run_case(n):
    temperatures = np.arange(300,1105,5)
    mch = ct.Solution('therm-data.xml')
    mch_mw = mch.molecular_weights[mch.species_index('mch')]
    mch.basis = 'molar'
    mch_cp = np.zeros(len(temperatures))
    Ti = 294.45
    tank_volume = 0.017561

    mass_fuel = 3.46
    mole_fuel = mass_fuel/mch_mw
    mole_o2 = mole_fuel*10.5
    mole_n2 = mole_fuel*12.25
    mole_ar = mole_fuel*71.75

    nom_o2_pres = mole_o2*ct.gas_constant*Ti/(1000*tank_volume)
    nom_n2_pres = mole_n2*ct.gas_constant*Ti/(1000*tank_volume)
    nom_ar_pres = mole_ar*ct.gas_constant*Ti/(1000*tank_volume)

    sigma_pressure = 346.6/2
    sigma_mass = 0.03
    # Convert the initial temperature to °C to match the spec.
    sigma_Ti = max(2.2,(Ti-273.15)*0.0075)/2

    Ti_dist = norm_dist(loc=Ti, scale=sigma_Ti)
    mass_dist = norm_dist(loc=mass_fuel, scale=sigma_mass)

    T0 = 308
    # Convert the initial temperature to °C to match the spec.
    sigma_T0 = max(2.2,(T0-273)*0.0075)/2
    T0_dist = norm_dist(loc=T0, scale=sigma_T0)

    P0 = 213755.753
    sigma_P0 = 346.6/2
    P0_dist = norm_dist(loc=P0, scale=sigma_P0)

    PC = 50.0054*1E5
    sigma_PC = 5000/2
    PC_dist = norm_dist(loc=PC, scale=sigma_PC)

    o2_dist = norm_dist(loc=nom_o2_pres, scale=sigma_pressure)
    o2_pres_rand = o2_dist.ppf(np.random.random_sample())
    n2_pressure = o2_pres_rand + nom_n2_pres
    n2_dist = norm_dist(loc=n2_pressure, scale=sigma_pressure)
    n2_pres_rand = n2_dist.ppf(np.random.random_sample())
    ar_pressure = n2_pres_rand + nom_ar_pres
    ar_dist = norm_dist(loc=ar_pressure, scale=sigma_pressure)
    ar_pres_rand = ar_dist.ppf(np.random.random_sample())
    mole_fuel_rand = mass_dist.ppf(np.random.random_sample())/mch_mw
    Ti_rand = Ti_dist.ppf(np.random.random_sample())
    T0_rand = T0_dist.ppf(np.random.random_sample())
    o2_pres_T0 = T0_rand*o2_pres_rand/Ti_rand
    n2_pres_T0 = T0_rand*n2_pres_rand/Ti_rand
    ar_pres_T0 = T0_rand*ar_pres_rand/Ti_rand

    mole_o2_rand = o2_pres_T0*tank_volume*1000/(ct.gas_constant*Ti_rand)
    mole_n2_rand = n2_pres_T0*tank_volume*1000/(ct.gas_constant*Ti_rand)
    mole_ar_rand = ar_pres_T0*tank_volume*1000/(ct.gas_constant*Ti_rand)
    total_moles = sum([mole_fuel_rand, mole_o2_rand, mole_n2_rand, mole_ar_rand])
    mole_fractions = 'mch:{mch},o2:{o2},n2:{n2},ar:{ar}'.format(
        mch=mole_fuel_rand/total_moles, o2=mole_o2_rand/total_moles,
        n2=mole_n2_rand/total_moles, ar=mole_ar_rand/total_moles)
    mch.TPX = None, None, mole_fractions
    
    for i, temp in enumerate(temperatures):
        mch.TP = temp, None
        mch_cp[i] = mch.cp/ct.gas_constant

    (mch_b, mch_a) = np.polyfit(temperatures, mch_cp, 1)
    P0_rand = P0_dist.ppf(np.random.random_sample())
    PC_rand = PC_dist.ppf(np.random.random_sample())
    mch_lam_rand = mch_b/mch_a * np.exp(mch_b*T0_rand/mch_a) * T0_rand * (PC_rand/P0_rand)**(1/mch_a)
    mch_TC_rand = np.real(mch_a * lambertw(mch_lam_rand)/mch_b)
    return mch_TC_rand


if __name__ == "__main__":
    start = time.time()
    # n is the number iterations to run
    n = 1000000

    # Set up a pool of processors to run in parallel
    pool = Pool(processes=20)

    # Run the analysis and get the result into a NumPy array.
    result = np.array(pool.map(run_case, range(n)))

    # Print the mean and twice the standard deviation to a file.
    with open('results.txt', 'a') as output:
        print(n, result.mean(), result.std()*2, file=output)
    print(time.time() - start)

    # Create and save the histogram data file. Compile the TeX file to make a
    # PDF figure of the histogram.
    hist, bin_edges = np.histogram(result, bins=100, density=True)
    np.savetxt('histogram/histogram.dat', np.vstack((np.insert(bin_edges, 0, result.mean()), np.insert(np.append(hist,0), 0, result.std()))).T)
    os.chdir('histogram')
    subprocess.call(['pdflatex', '-interaction=batchmode', 'histogram'])
    # print(result)
    # n, bins, patches = plt.hist(result, 100, normed=1, facecolor='green', alpha=0.75)

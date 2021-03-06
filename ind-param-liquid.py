#! /usr/bin/env python3
"""
ind-param-liquid.py

This script runs the independent parameters uncertainty analysis for
a liquid fuel in the University of Connecticut RCM. This script is
associated with the work "On the Uncertainty of Temperature Estimation
in a Rapid Compression Machine" by Bryan W. Weber, Chih-Jen Sung, and
Michael W. Renfro, Combustion and Flame, DOI:
10.1016/j.combustflame.2015.03.001. This script is
licensed according to the LICENSE file available in the repository
associated in the paper.

The most recent version of this code is available on GitHub
at https://github.com/bryanwweber/rcm-temperature-uncertainty

Please email weber@engineer.uconn.edu with any questions.
"""

# System library imports
import sys

try:
    from scipy.special import lambertw
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

if __name__ == "__main__":
    # Set the parameters to be studied so that we can use a loop
    P0s = [1.8794E5, 4.3787E5, 3.9691E5, 4.3635E5, 1.9118E5, 4.3987E5]
    T0s = [308]*6
    PCs = [50.0135E5, 49.8629E5, 50.0485E5, 49.6995E5, 49.8254E5, 50.0202E5]
    mfuels = [3.43, 3.48, 3.49, 3.53, 3.53, 3.69]
    Tas = [21.7, 21.7, 22.0, 22.1, 21.7, 20.0, ]
    cases = ['a', 'b', 'c', 'd', 'e', 'f']

    # Set the string of the fuel.
    fuel = 'mch'

    # Set the mixtures to study. These are described in the paper
    mix1 = [10.5, 12.25, 71.75]
    mix2 = [21.0, 00.00, 73.50]
    mix3 = [07.0, 16.35, 71.15]

    # Use the larger mixing tank for reactive mixtures.
    nom_tank_volume = 0.01660

    # Set the uncertainties of the parameters
    delta_volume = 0.00001  # m**3
    delta_fuel_mass = 0.03  # grams
    delta_Pi = 346.6/2  # Pa
    delta_P0 = 346.6/2  # Pa
    delta_PC = 5000/2  # Pa

    # Convert the gas constant from Cantera to J/mol-K
    R_u = ct.gas_constant/1000

    # Initialize the delta_TC and T_C arrays. delta_TC is the
    # uncertainty of T_C.
    delta_TC = np.zeros(len(cases))
    T_C = np.zeros(len(cases))

    # Initialize the Cantera Solution and set molar as the basis for
    # the properties.
    gas = ct.Solution('therm-data.xml')
    fuel_mw = gas.molecular_weights[gas.species_index(fuel)]
    gas.basis = 'molar'

    temperatures = np.arange(300, 1101, 1)
    gas_cp = np.zeros(len(temperatures))
    fuel_cp = np.zeros(len(temperatures))
    o2_cp = np.zeros(len(temperatures))
    n2_cp = np.zeros(len(temperatures))
    ar_cp = np.zeros(len(temperatures))

    # In this method, the uncertainty of each gaseous component
    # is considered separately, so compute the specific heat
    # for each component
    for i, temperature in enumerate(temperatures):
        gas.TPX = temperature, None, '{fuel_name}:1'.format(fuel_name=fuel)
        fuel_cp[i] = gas.cp/ct.gas_constant
        gas.TPX = None, None, 'o2:1'
        o2_cp[i] = gas.cp/ct.gas_constant
        gas.TPX = None, None, 'n2:1'
        n2_cp[i] = gas.cp/ct.gas_constant
        gas.TPX = None, None, 'ar:1'
        ar_cp[i] = gas.cp/ct.gas_constant

    for j, case in enumerate(cases):
        # Each case is associated with a particular mixture in the
        # paper. Set which mixture to use here.
        if case == 'a' or case == 'b':
            mix = mix1
        elif case == 'c' or case == 'd':
            mix = mix2
        else:
            mix = mix3

        P0 = P0s[j]
        T0 = T0s[j]
        PC = PCs[j]
        nom_mass_fuel = mfuels[j]
        Ta = Tas[j] + 273.15

        # Compute the uncertainty of the initial temperature. For the
        # assumption of normally distributed uncertainty, divide by
        # two. For uniformly distributed uncertainty, divide by the
        # square root of three. For the triangular distribution, divide
        # by the square root of six.
        delta_Ta = max(2.2, (Ta - 273.15)*0.0075)/2  # degrees C
        # delta_Ta = max(2.2, (Ta - 273.15)*0.0075)/np.sqrt(3)  # degrees C
        # delta_Ta = max(2.2, (Ta - 273.15)*0.0075)/np.sqrt(6)  # degrees C
        delta_T0 = max(2.2, (T0 - 273)*0.0075)/2  # degrees C
        # delta_T0 = max(2.2, (T0 - 273)*0.0075)/np.sqrt(3)  # degrees C
        # delta_T0 = max(2.2, (T0 - 273)*0.0075)/np.sqrt(6)  # degrees C

        # Compute the nominal moles of fuel and corresponding nominal required
        # number of moles of the gases.
        nom_mole_fuel = nom_mass_fuel/fuel_mw
        nom_mole_o2 = nom_mole_fuel*mix[0]
        nom_mole_n2 = nom_mole_fuel*mix[1]
        nom_mole_ar = nom_mole_fuel*mix[2]

        # Compute the mole fractions of each component and set the state of the
        # Cantera solution.
        total_moles = sum([nom_mole_fuel, nom_mole_o2, nom_mole_n2, nom_mole_ar])
        mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
            fuel_name=fuel, fuel_mole=nom_mole_fuel/total_moles, o2=nom_mole_o2/total_moles,
            n2=nom_mole_n2/total_moles, ar=nom_mole_ar/total_moles)
        gas.TPX = None, None, mole_fractions

        # Compute the total specific heat curve.
        for i, temperature in enumerate(temperatures):
            gas.TP = temperature, None
            gas_cp[i] = gas.cp/ct.gas_constant

        # Calculate the uncertainty in the number of moles of fuel.
        delta_fuel_moles = delta_fuel_mass/fuel_mw

        # Calculate the nominal pressure required for each gas to match the
        # desired molar proportions.
        nom_o2_pres = nom_mole_o2*R_u*Ta/(nom_tank_volume)
        nom_n2_pres = nom_mole_n2*R_u*Ta/(nom_tank_volume)
        nom_ar_pres = nom_mole_ar*R_u*Ta/(nom_tank_volume)

        # Calculate the square of the uncertainty in the number of
        # moles of each gaseous component.
        delta_o2_2 = ((nom_tank_volume/(R_u*Ta)*delta_Pi)**2 +
                      (nom_o2_pres/(R_u*Ta)*delta_volume)**2 +
                      (-nom_o2_pres*nom_tank_volume/(R_u*Ta**2)*delta_Ta)**2)

        if nom_n2_pres > 0:
            delta_n2_2 = ((nom_tank_volume/(R_u*Ta)*delta_Pi)**2 +
                          (nom_n2_pres/(R_u*Ta)*delta_volume)**2 +
                          (-nom_n2_pres*nom_tank_volume/(R_u*Ta**2)*delta_Ta)**2)
        else:
            delta_n2_2 = 0

        delta_ar_2 = ((nom_tank_volume/(R_u*Ta)*delta_Pi)**2 +
                      (nom_ar_pres/(R_u*Ta)*delta_volume)**2 +
                      (-nom_ar_pres*nom_tank_volume/(R_u*Ta**2)*delta_Ta)**2)

        # Calculate the square of the uncertainty in the mole fraction
        # of each component.
        delta_X_fuel_2 = ((sum([nom_mole_o2, nom_mole_n2, nom_mole_ar])/total_moles**2)*delta_fuel_moles)**2
        fuel_sum = 0
        for delta in [delta_o2_2, delta_n2_2, delta_ar_2]:
            fuel_sum += ((-nom_mole_fuel/total_moles**2)**2)*delta
        delta_X_fuel_2 += fuel_sum

        delta_X_o2_2 = ((sum([nom_mole_fuel, nom_mole_n2, nom_mole_ar])/total_moles**2)**2)*delta_o2_2
        o2_sum = 0
        for delta in [delta_fuel_moles**2, delta_n2_2, delta_ar_2]:
            o2_sum += ((-nom_mole_o2/total_moles**2)**2)*delta
        delta_X_o2_2 += o2_sum

        delta_X_n2_2 = ((sum([nom_mole_fuel, nom_mole_o2, nom_mole_ar])/total_moles**2)**2)*delta_n2_2
        n2_sum = 0
        for delta in [delta_fuel_moles**2, delta_o2_2, delta_ar_2]:
            n2_sum += ((-nom_mole_n2/total_moles**2)**2)*delta
        delta_X_n2_2 += n2_sum

        delta_X_ar_2 = ((sum([nom_mole_fuel, nom_mole_n2, nom_mole_o2])/total_moles**2)**2)*delta_ar_2
        ar_sum = 0
        for delta in [delta_fuel_moles**2, delta_n2_2, delta_o2_2]:
            ar_sum += ((-nom_mole_ar/total_moles**2)**2)*delta
        delta_X_ar_2 += ar_sum

        delta_Cp_2 = np.zeros(len(temperatures))
        for i in range(len(temperatures)):
            for cp, delta in zip([fuel_cp[i], o2_cp[i], n2_cp[i], ar_cp[i]],
                                 [delta_X_fuel_2, delta_X_o2_2, delta_X_n2_2,
                                  delta_X_ar_2]):
                delta_Cp_2[i] += delta*cp**2

        # Compute the slope and y-intercept and their uncertainties by the
        # York procedure.
        omega_Cp = 1/delta_Cp_2
        T_bar = np.sum(omega_Cp*temperatures)/np.sum(omega_Cp)
        Cp_bar = np.sum(omega_Cp*gas_cp)/np.sum(omega_Cp)
        F = temperatures - T_bar
        G = gas_cp - Cp_bar

        b_guess = np.zeros(2)
        (b_guess[0], dummy) = np.polyfit(temperatures, gas_cp, 1)
        beta = F + omega_Cp*b_guess[0]*G
        err = 1
        while err > 1E-15:
            b_guess[1] = np.sum(omega_Cp*beta*G)/np.sum(omega_Cp*beta*F)
            err = (b_guess[1] - b_guess[0])/b_guess[0]
            b_guess[0] = b_guess[1]
            beta = F + omega_Cp*b_guess[1]*G

        b = b_guess[1]
        a = Cp_bar - b*T_bar
        t = T_bar + beta
        t_bar = np.sum(omega_Cp*t)/np.sum(omega_Cp)
        f = t - t_bar
        delta_b_2 = 1/np.sum(omega_Cp*f**2)
        delta_b = np.sqrt(delta_b_2)
        delta_a_2 = 1/np.sum(omega_Cp) + t_bar**2*delta_b_2
        delta_a = np.sqrt(delta_a_2)

        # Compute the partial derivatives in the uncertainty formula.
        D = np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))

        partial_PC = D/(b*PC*(D+1))
        partial_P0 = -D/(b*P0*(D+1))
        partial_T0 = ((a + b*T0)*D)/(b*T0*(D+1))
        partial_a = (-D*(b*T0 + np.log(PC/P0) - a*D))/(a*b*(D+1))
        partial_b = (D*(b*T0 - a*D))/(b**2*(D+1))

        # Compute the uncertainty in T_C.
        delta_TC_2 = 0
        for partial, delta in zip([partial_PC, partial_P0, partial_T0, partial_a, partial_b],
                                  [delta_PC, delta_P0, delta_T0, delta_a, delta_b]):
            delta_TC_2 += (partial*delta)**2

        delta_TC[j] = 2*np.sqrt(delta_TC_2)
        T_C[j] = a*np.real(lambertw(b/a*np.exp(b*T0/a)*T0*(PC/P0)**(1/a)))/b

    # Set the format of screen output and print to screen.
    np.set_printoptions(formatter={'all':lambda x: '{0:.2f}'.format(x)})
    print('δ_T_C: ', delta_TC)
    print('T_C: ', T_C)

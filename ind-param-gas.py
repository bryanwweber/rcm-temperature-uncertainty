#! /usr/bin/env python3
"""
ind-param-gas.py

This script runs the independent parameters uncertainty analysis for
a gaseous fuel in the University of Connecticut RCM. This script is
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
    P0s = [0.3613E5, 0.5612E5, 1.2547E5, 1.8601E5, 1.6951E5, 1.7138E5,
           0.3763E5, 0.5619E5]
    T0s = [413, 413, 413, 398, 398, 358, 413, 413]
    PCs = [10.0918E5, 10.1670E5, 40.6010E5, 40.5571E5, 40.7850E5,
           40.5031E5, 10.1301E5, 10.0784E5]
    cases = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # Set the string of the fuel.
    fuel = 'c3h6'

    # Set the mixtures to study. These are described in the paper
    mix1 = {'o2': 257, 'n2': 739, 'ar': 1223, 'fuel': 1280}
    mix2 = {'ar': 0, 'n2': 1413, 'o2': 1473, 'fuel': 1500}
    mix3 = {'ar': 0, 'n2': 1607, 'o2': 1675, 'fuel': 1706}
    mix4 = {'ar': 0, 'n2': 1536, 'o2': 1752, 'fuel': 1800}
    mix5 = {'ar': 0, 'n2': 1365, 'o2': 1557, 'fuel': 1600}
    mix6 = {'o2': 216, 'n2': 984, 'ar': 1752, 'fuel': 1800}

    # Set the uncertainties of the parameters
    delta_Pi = 346.6/2  # Pa
    delta_P0 = 346.6/2  # Pa
    delta_PC = 5000/2  # Pa

    # Convert pressures in Torr to Pa
    torr_to_pa = ct.one_atm/760

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
        elif case == 'c':
            mix = mix2
        elif case == 'd':
            mix = mix3
        elif case == 'e':
            mix = mix4
        elif case == 'f':
            mix = mix5
        elif case == 'g' or case == 'h':
            mix = mix6

        P0 = P0s[j]
        T0 = T0s[j]
        PC = PCs[j]

        # Compute the partial pressures of each component and the
        # uncertainty of the mole fraction of each component. The _2
        # indicates that the variance is being calculated.
        if mix['ar'] > 0:
            total_pressure = mix['fuel']*torr_to_pa
            fuel_par_pres = (mix['fuel'] - mix['ar'])*torr_to_pa
            ar_par_pres = (mix['ar'] - mix['n2'])*torr_to_pa
            n2_par_pres = (mix['n2'] - mix['o2'])*torr_to_pa
            o2_par_pres = mix['o2']*torr_to_pa

            # Fuel
            delta_X_fuel_2 = ((sum([ar_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_fuel_2 += 3*((-fuel_par_pres/total_pressure**2)*delta_Pi)**2
            # Oxygen
            delta_X_o2_2 = ((sum([ar_par_pres, n2_par_pres, fuel_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 += 3*((-o2_par_pres/total_pressure**2)*delta_Pi)**2
            # Nitrogen
            delta_X_n2_2 = ((sum([ar_par_pres, fuel_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 += 3*((-n2_par_pres/total_pressure**2)*delta_Pi)**2
            # Argon
            delta_X_ar_2 = ((sum([fuel_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_ar_2 += 3*((-ar_par_pres/total_pressure**2)*delta_Pi)**2
        else:
            total_pressure = mix['fuel']*torr_to_pa
            fuel_par_pres = (mix['fuel'] - mix['o2'])*torr_to_pa
            o2_par_pres = (mix['o2'] - mix['n2'])*torr_to_pa
            n2_par_pres = mix['n2']*torr_to_pa
            ar_par_pres = 0

            # Fuel
            delta_X_fuel_2 = ((sum([ar_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_fuel_2 += 2*((-fuel_par_pres/total_pressure**2)*delta_Pi)**2
            # Oxygen
            delta_X_o2_2 = ((sum([ar_par_pres, n2_par_pres, fuel_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 += 2*((-o2_par_pres/total_pressure**2)*delta_Pi)**2
            # Nitrogen
            delta_X_n2_2 = ((sum([ar_par_pres, fuel_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 += 2*((-n2_par_pres/total_pressure**2)*delta_Pi)**2

        # Compute the uncertainty of the initial temperature. For the
        # assumption of normally distributed uncertainty, divide by
        # two. For uniformly distributed uncertainty, divide by the
        # square root of three. For the triangular distribution, divide
        # by the square root of six.
        delta_T0 = max(2.2, (T0 - 273)*0.0075)/4  # degrees C
        # delta_T0 = max(2.2, (T0 - 273)*0.0075)/np.sqrt(3)  # degrees C
        # delta_T0 = max(2.2, (T0 - 273)*0.0075)/np.sqrt(6)  # degrees C

        mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
            fuel_name=fuel, fuel_mole=fuel_par_pres/total_pressure, o2=o2_par_pres/total_pressure,
            n2=n2_par_pres/total_pressure, ar=ar_par_pres/total_pressure)
        gas.TPX = None, None, mole_fractions

        # Compute the total specific heat curve.
        for i, temperature in enumerate(temperatures):
            gas.TP = temperature, None
            gas_cp[i] = gas.cp/ct.gas_constant

        # Compute the uncertainty of the specific heat
        delta_Cp_2 = np.zeros(len(temperatures))
        for i in range(len(temperatures)):
            for cp, delta in zip([fuel_cp[i], o2_cp[i], n2_cp[i], ar_cp[i]],
                                 [delta_X_fuel_2, delta_X_o2_2, delta_X_n2_2,
                                  delta_X_ar_2]):
                delta_Cp_2[i] += delta*cp**2

        # Compute the slope, y-intercept, and their uncertainties by the
        # York procedure
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
        T_C[j] = a*np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))/b

    np.set_printoptions(formatter={'all':lambda x: '{0:.2f}'.format(x)})
    print('δ_T_C: ', delta_TC)
    print('T_C: ', T_C)

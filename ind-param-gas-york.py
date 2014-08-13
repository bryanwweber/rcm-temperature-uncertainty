#! /usr/bin/env python3
from __future__ import division, print_function
from scipy.special import lambertw
import cantera as ct
import numpy as np

if __name__ == "__main__":
    # Set the parameters to be studied so that we can use a loop
    P0s = [0.3613E5, 0.5612E5, 1.2547E5, 1.8601E5, 1.6951E5, 1.7138E5, 0.3763E5, 0.5619E5,]
    T0s = [413, 413, 413, 398, 398, 358, 413, 413,]
    PCs = [10.0918E5, 10.1670E5, 40.6010E5, 40.5571E5, 40.7850E5, 40.5031E5, 10.1301E5, 10.0784E5,]
    cases = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # Set the string of the fuel. Possible values with the distributed
    # therm-data.xml are 'mch', 'nc4h9oh', 'sc4h9oh', 'tc4h9oh', 'ic4h9oh',
    # 'ic5h11oh', and 'c3h6'
    fuel = 'c3h6'

    # Set the mixtures to study
    mix1 = {'o2':257, 'n2':739, 'ar':1223, 'fuel':1280,}
    mix2 = {'ar':0, 'n2':1413, 'o2':1473, 'fuel':1500,}
    mix3 = {'ar':0, 'n2':1607, 'o2':1675, 'fuel':1706,}
    mix4 = {'ar':0, 'n2':1536, 'o2':1752, 'fuel':1800,}
    mix5 = {'ar':0, 'n2':1365, 'o2':1557, 'fuel':1600,}
    mix6 = {'o2':216, 'n2':984, 'ar':1752, 'fuel':1800,}

    # Set the uncertainties of the parameters
    delta_Pi = 346.6/2 # Pa
    delta_P0 = 346.6/2 # Pa
    delta_PC = 5000/2 # Pa

    # Convert the gas constant from Cantera to J/mol-K
    R = ct.gas_constant/1000
    
    # Convert pressures in Torr to Pa
    torr_to_pa = ct.one_atm/760

    # Initialize the delta_TC and TC arrays
    delta_TC = np.zeros(len(cases))
    TC = np.zeros(len(cases))

    # Initialize the Cantera Solution
    gas = ct.Solution('therm-data.xml')
    fuel_mw = gas.molecular_weights[gas.species_index(fuel)]
    gas.basis = 'molar'

    temperatures = np.arange(300,1101,1)
    gas_cp = np.zeros(len(temperatures))
    fuel_cp = np.zeros(len(temperatures))
    o2_cp = np.zeros(len(temperatures))
    n2_cp = np.zeros(len(temperatures))
    ar_cp = np.zeros(len(temperatures))
    f = [11, 3,]

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

        if mix['ar'] > 0:
            total_pressure = mix['fuel']*torr_to_pa
            fuel_par_pres = (mix['fuel'] - mix['ar'])*torr_to_pa
            ar_par_pres = (mix['ar'] - mix['n2'])*torr_to_pa
            n2_par_pres = (mix['n2'] - mix['o2'])*torr_to_pa
            o2_par_pres = mix['o2']*torr_to_pa

            delta_X_fuel_2 = ((sum([ar_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_fuel_2 += 3*((-fuel_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 = ((sum([ar_par_pres, n2_par_pres, fuel_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 += 3*((-o2_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 = ((sum([ar_par_pres, fuel_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 += 3*((-n2_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_ar_2 = ((sum([fuel_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_ar_2 += 3*((-ar_par_pres/total_pressure**2)*delta_Pi)**2
        else:
            total_pressure = mix['fuel']*torr_to_pa
            fuel_par_pres = (mix['fuel'] - mix['o2'])*torr_to_pa
            o2_par_pres = (mix['o2'] - mix['n2'])*torr_to_pa
            n2_par_pres = mix['n2']*torr_to_pa
            ar_par_pres = 0

            delta_X_fuel_2 = ((sum([ar_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_fuel_2 += 2*((-fuel_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 = ((sum([ar_par_pres, n2_par_pres, fuel_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 += 2*((-o2_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 = ((sum([ar_par_pres, fuel_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 += 2*((-n2_par_pres/total_pressure**2)*delta_Pi)**2

        delta_T0 = max(2.2, (T0 - 273)*0.0075)/2 # °C

        mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
            fuel_name=fuel, fuel_mole=fuel_par_pres/total_pressure, o2=o2_par_pres/total_pressure,
            n2=n2_par_pres/total_pressure, ar=ar_par_pres/total_pressure)
        gas.TPX = None, None, mole_fractions

        # Compute the total specific heat at the end points of the curve to be fit
        for i, temperature in enumerate(temperatures):
            gas.TP = temperature, None
            gas_cp[i] = gas.cp/ct.gas_constant

        delta_Cp_2 = np.zeros(len(temperatures))
        for i in range(len(temperatures)):
            for cp, delta in zip([fuel_cp[i], o2_cp[i], n2_cp[i], ar_cp[i]],
                                 [delta_X_fuel_2, delta_X_o2_2, delta_X_n2_2,
                                    delta_X_ar_2]):
                delta_Cp_2[i] += delta*cp**2

        omega_Cp = 1/delta_Cp_2
        T_bar = np.sum(omega_Cp*temperatures)/np.sum(omega_Cp)
        Cp_bar = np.sum(omega_Cp*gas_cp)/np.sum(omega_Cp)
        F = temperatures - T_bar
        G = gas_cp - Cp_bar

        # Compute the slope and y-intercept and their uncertainties by the
        # York procedure
        b_guess = np.zeros(2)
        (b_guess[0], _) = np.polyfit(temperatures, gas_cp, 1)
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
        delta_a_2 = 1/np.sum(omega_Cp) + t_bar**2*delta*b*2
        delta_a = np.sqrt(delta_a_2)

        D = np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))

        partial_PC = D/(b*PC*(D+1))
        partial_P0 = -D/(b*P0*(D+1))
        partial_T0 = ((a + b*T0)*D)/(b*T0*(D+1))
        partial_a = (-D*(b*T0 + np.log(PC/P0) - a*D))/(a*b*(D+1))
        partial_b = (D*(b*T0 - a*D))/(b**2*(D+1))

        delta_TC_2 = 0
        for partial, delta in zip([partial_PC, partial_P0, partial_T0, partial_a, partial_b],
                                  [delta_PC, delta_P0, delta_T0, delta_a, delta_b]):
            delta_TC_2 += (partial*delta)**2

        delta_TC[j] = 2*np.sqrt(delta_TC_2)
        TC[j] = a*np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))/b

    print(delta_TC, TC)

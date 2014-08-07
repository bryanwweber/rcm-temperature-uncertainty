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
    mix2 = {'o2':216, 'n2':984, 'ar':1752, 'fuel':1800,}
    mix3 = {'ar':0, 'n2':1536, 'o2':1752, 'fuel':1800,}
    mix4 = {'ar':0, 'n2':1365, 'o2':1557, 'fuel':1600,}
    mix5 = {'ar':0, 'n2':1413, 'o2':1473, 'fuel':1500,}
    mix6 = {'ar':0, 'n2':1607, 'o2':1675, 'fuel':1706,}

    # Set the uncertainties of the parameters
    delta_Pi = 346.6 # Pa
    delta_P0 = 346.6 # Pa
    delta_PC = 5000 # Pa

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

    temperatures = [300, 1100,]
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
        elif case == 'g' or case == 'h':
            mix = mix2
        elif case == 'f':
            mix = mix3
        elif case == 'e':
            mix = mix4
        elif case == 'c':
            mix = mix5
        elif case == 'd':
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

            delta_X_fuel_2 = ((sum([ar_par_pres, n2_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_fuel_2 += 2*((-fuel_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 = ((sum([ar_par_pres, n2_par_pres, fuel_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_o2_2 += 2*((-o2_par_pres/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 = ((sum([ar_par_pres, fuel_par_pres, o2_par_pres])/total_pressure**2)*delta_Pi)**2
            delta_X_n2_2 += 2*((-n2_par_pres/total_pressure**2)*delta_Pi)**2

        delta_T0 = max(2.2, (T0 - 273)*0.0075) # °C

        mole_fractions = '{fuel_name}:{fuel_mole},o2:{o2},n2:{n2},ar:{ar}'.format(
            fuel_name=fuel, fuel_mole=fuel_par_pres/total_pressure, o2=o2_par_pres/total_pressure,
            n2=n2_par_pres/total_pressure, ar=ar_par_pres/total_pressure)
        gas.TPX = None, None, mole_fractions

        # Compute the total specific heat at the end points of the curve to be fit
        for i, temperature in enumerate(temperatures):
            gas.TP = temperature, None
            gas_cp[i] = gas.cp/ct.gas_constant

        # Compute the slope and y-intercept based on the end points
        b = (gas_cp[1] - gas_cp[0])/(temperatures[1] - temperatures[0])
        a = (f[1]*gas_cp[1] - f[0]*gas_cp[0])/(f[1] - f[0])

        D = np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))

        partial_PC = D/(b*PC*(D+1))
        partial_P0 = -D/(b*P0*(D+1))
        partial_T0 = ((a + b*T0)*D)/(b*T0*(D+1))
        partial_a = (-D*(b*T0 + np.log(PC/P0) - a*D))/(a*b*(D+1))
        partial_b = (D*(b*T0 - a*D))/(b**2*(D+1))

        delta_Cp = np.zeros(len(temperatures))
        for i in range(len(temperatures)):
            for cp, delta in zip([fuel_cp[i], o2_cp[i], n2_cp[i], ar_cp[i]],
                                 [delta_X_fuel_2, delta_X_o2_2, delta_X_n2_2,
                                    delta_X_ar_2]):
                delta_Cp[i] += delta*cp**2

        partial_b_Cp = np.array([-1/(temperatures[1] - temperatures[0]), 1/(temperatures[1] - temperatures[0])])
        delta_b_2 = (partial_b_Cp[0]*delta_Cp[0])**2 + (partial_b_Cp[1]*delta_Cp[1])**2
        delta_b = np.sqrt(delta_b_2)
        partial_a_Cp = np.array([-f[0]/(f[1] - f[0]), f[1]/(f[1] - f[0])])
        delta_a_2 = (partial_a_Cp[0]*delta_Cp[0])**2 + (partial_a_Cp[1]*delta_Cp[1])**2
        delta_a = np.sqrt(delta_a_2)

        delta_TC_2 = 0
        for partial, delta in zip([partial_PC, partial_P0, partial_T0, partial_a, partial_b],
                                  [delta_PC, delta_P0, delta_T0, delta_a, delta_b]):
            delta_TC_2 += (partial*delta)**2

        delta_TC[j] = np.sqrt(delta_TC_2)
        TC[j] = a*np.real(lambertw(b/a*np.exp((b*T0)/a)*T0*(PC/P0)**(1/a)))/b

    print(delta_TC, TC)

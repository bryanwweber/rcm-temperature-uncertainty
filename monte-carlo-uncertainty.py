from scipy.special import lambertw
from scipy.stats import norm as norm_dist
import cantera as ct
import numpy as np


mch = ct.Solution('therm-data.xml')
nbuoh = ct.Solution('therm-data.xml')
propene = ct.Solution('therm-data.xml')
ipeoh = ct.Solution('therm-data.xml')

mch.TPX = None, None, 'mch:0.0105,o2:0.2199,ar:0.7696'
nbuoh.TPX = None, None, 'nc4h9oh:0.0676,o2:0.203,n2:0.7294'
ipeoh.TPX = None, None, 'ic5h11oh:0.0531,o2:0.1989,n2:0.7480'
propene.TPX = None, None, 'c3h6:0.0854,o2:0.1921,n2:0.7225'

mch.basis = 'molar'
nbuoh.basis = 'molar'
ipeoh.basis = 'molar'
propene.basis = 'molar'

temperatures = np.arange(300,1105,5)

mch_cp = np.zeros(len(temperatures))
nbuoh_cp = np.zeros(len(temperatures))
ipeoh_cp = np.zeros(len(temperatures))
propene_cp = np.zeros(len(temperatures))

for i, temp in enumerate(temperatures):
    mch.TP = temp, None
    nbuoh.TP = temp, None
    ipeoh.TP = temp, None
    propene.TP = temp, None
    mch_cp[i] = mch.cp/8314.4621
    nbuoh_cp[i] = nbuoh.cp/8314.4621
    ipeoh_cp[i] = ipeoh.cp/8314.4621
    propene_cp[i] = propene.cp/8314.4621

(mch_b, mch_a), mch_var = np.polyfit(temperatures, mch_cp, 1, cov=True)[:2]
(nbuoh_b, nbuoh_a), nbuoh_var = np.polyfit(temperatures, nbuoh_cp, 1, cov=True)[:2]
(ipeoh_b, ipeoh_a), ipeoh_var = np.polyfit(temperatures, ipeoh_cp, 1, cov=True)[:2]
(propene_b, propene_a), propene_var = np.polyfit(temperatures, propene_cp, 1, cov=True)[:2]

# Return the R^2 values for the linear fits
# (mch_b, mch_a), mch_resid = np.polyfit(temperatures, mch_cp, 1, full=True)[:2]
# mch_r2 = 1 - mch_resid/(mch_cp.size*mch_cp.var())
# nbuoh_r2 = 1 - nbuoh_resid/(nbuoh_cp.size*nbuoh_cp.var())
# ipeoh_r2 = 1 - ipeoh_resid/(ipeoh_cp.size*ipeoh_cp.var())
# propene_r2 = 1 - propene_resid/(propene_cp.size*propene_cp.var())

# Assume that the uncertainty quoted by the manufacturer is ±2σ. scale in the
# normal distribution provided by scipy is the standard deviation.

T0 = 360
# Convert the initial temperature to °C to match the spec.
sigma_T0 = max(2.2,(T0-273)*0.0075)/2
T0_dist = norm_dist(loc=T0, scale=sigma_T0)

P0 = 0.5*1E5
sigma_P0 = 346.6/2
P0_dist = norm_dist(loc=P0, scale=sigma_P0)

PC = 15*1E5
sigma_PC = 5000/2
PC_dist = norm_dist(loc=PC, scale=sigma_PC)

mch_lam = mch_b/mch_a * np.exp(mch_b*T0/mch_a) * T0 * (PC/P0)**(1/mch_a)
mch_TC = np.real(mch_a * lambertw(mch_lam)/mch_b)
# nbuoh_lam = nbuoh_b/nbuoh_a * np.exp(nbuoh_b*T0/nbuoh_a) * T0 * (PC/P0)**(1/nbuoh_a)
# nbuoh_TC = np.real(nbuoh_a * lambertw(nbuoh_lam)/nbuoh_b)
# ipeoh_lam = ipeoh_b/ipeoh_a * np.exp(ipeoh_b*T0/ipeoh_a) * T0 * (PC/P0)**(1/ipeoh_a)
# ipeoh_TC = np.real(ipeoh_a * lambertw(ipeoh_lam)/ipeoh_b)
# propene_lam = propene_b/propene_a * np.exp(propene_b*T0/propene_a) * T0 * (PC/P0)**(1/propene_a)
# propene_TC = np.real(propene_a * lambertw(propene_lam)/propene_b)
# print(mch_TC, nbuoh_TC, ipeoh_TC, propene_TC)

T0_rand = T0_dist.ppf(np.random.random_sample(10000000))
P0_rand = P0_dist.ppf(np.random.random_sample(10000000))
PC_rand = PC_dist.ppf(np.random.random_sample(10000000))
mch_lam_rand = mch_b/mch_a * np.exp(mch_b*T0_rand/mch_a) * T0_rand * (PC_rand/P0_rand)**(1/mch_a)
mch_TC_rand = np.real(mch_a * lambertw(mch_lam_rand)/mch_b)

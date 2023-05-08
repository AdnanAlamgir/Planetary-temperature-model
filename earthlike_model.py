import numpy as np, matplotlib.pyplot as plt, pandas as pd, pylab
from scipy.optimize import fsolve
plt.rcParams.update({'font.size': 10})

L_sun = 3.828e26
R_sun = 6.96e8
a_earth = 1.496e11
R_earth = 6.371e6
M_earth = 5.97e24
A = 0.3
P_s = 10**5
RH_d = 0.8
RH_n = 0.6
z_o = 1000
z_a = 1000 
t_o = (1 / (100*365 * 24 * 3600))
t_a = (1 / (30 * 24 * 3600))

g = 9.8
sb = 5.67e-8 
L_v = 2.25e6
R_w = 462 
R_a = 287

kappa = 1000

C_pw = 4182
rho_w = 1000
C_pa = 1000

def fn(x):
    T_sd = x[0] 
    T_ad = x[1] 
    T_sn = x[2]
    T_an = x[3]

    P_z = P_s * np.exp((-z_a * g)/(R_a * T_ad))


    ## equation for r using classius-clapyron
    q_d = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_ad))
    r_d = 0.622 * (q_d / P_z) * RH_d
    em_d = 1 - np.exp(-kappa * r_d)
    em_d = 0.961


    q_n = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_an))
    r_n = 0.622 * (q_n / P_z) * RH_n
    em_n = 1 - np.exp(-kappa * r_n)
    em_n = 0.58


    q_sd = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_sd))
    r_sd = 0.622 * (q_sd / P_z) * RH_n


    A = 0.3

    #ocean and atmospheric heat transport
    F_o = C_pw * rho_w * z_o * t_o * (T_sd - T_sn)
    F_a = ((C_pa * P_z * t_a)/g) * (T_ad - T_an)

    f_1 = 0.5 * ((L_sun) / (4 * np.pi * (a_earth)**2)) * (1 - A) + em_d * sb * T_ad**4 - sb * T_sd**4 - F_o
    f_2 = sb * em_d * T_sd**4 - 2 * sb * em_d * T_ad**4 - F_a
    f_3 = em_n * sb * T_an**4 + F_o - sb * T_sn**4
    f_4 = sb * em_d * T_sn**4 + F_a - 2 * sb * em_n * T_an**4
    return f_1, f_2, f_3, f_4

sol3 = fsolve(fn, [300, 300, 200, 200])
print(sol3)

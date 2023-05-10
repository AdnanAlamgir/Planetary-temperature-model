import numpy as np, matplotlib.pyplot as plt, pandas as pd, pylab
from scipy.optimize import fsolve
from random import randint

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
z_a = 1000 # used in the moist static energy equation
t_o = (1 / (100*365 * 24 * 3600))
t_a = (1 / (30 * 24 * 3600))
g = 9.8

#Other constants
sb = 5.67e-8 
L_v = 2.25e6
R_w = 462 
R_a = 287
kappa = 1000
C_pw = 4182
rho_w = 1000
C_pa = 1000


dfinit = pd.read_csv('PS_2023.02.16_05.25.17.csv', usecols=(6, 10, 14, 18, 22), dtype=float, skiprows=47)
repvals = [0, 3, 4, 7, 8, 10, 19, 21, 29, 30, 31, 32, 34, 35, 36, 39, 41, 42, 43, 45, 46]
df = dfinit.drop(dfinit.index[repvals])
print(df)

a_plc, R_plc, M_plc, T_star, R_star = df['pl_orbsmax'].tolist(), df['pl_rade'].tolist(), df['pl_bmasse'].tolist(), df['st_teff'].tolist(), df['st_rad'].tolist()

plinit = pd.read_csv('PS_2023.02.16_05.25.17.csv', usecols=[0], dtype=str, skiprows=47)
plname = plinit['pl_name'].tolist()
plname2 = [plname[pl] for pl in range(50) if pl not in repvals]
headings = df.columns.tolist()

T_snlist, T_sdlist, T_anlist, T_adlist = [], [], [], []
Tsols = []
A = 0.3

for i in range(len(a_plc)):

  sol7b = [0, 0, 0, 0]
  g = (6.673e-11 * (M_plc[i] * M_earth))/(R_plc[i] * R_earth)**2
  L_star = 4 * np.pi * (R_star[i] * R_sun)**2 * sb * T_star[i]**4


  def f7(x):
    T_sd = x[0] 
    T_ad = x[1] 
    T_sn = x[2]
    T_an = x[3]

    P_z = P_s * np.exp((-z_a * g)/(R_a * T_ad))

    ## equation for r using classius-clapyron
    q_d = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_ad))
    r_d = 0.622 * (q_d / P_z) * RH_d
    em_d = 1 - np.exp(-kappa * r_d)

    q_n = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_an))
    r_n = 0.622 * (q_n / P_z) * RH_n
    em_n = 1 - np.exp(-kappa * r_n)

    q_sd = 611 * np.exp((L_v / R_w) * (1/273 - 1/T_sd))
    r_sd = 0.622 * (q_sd / P_z) * RH_n
    
    #ocean and atmospheric heat transport
    F_o = C_pw * rho_w * z_o * t_o * (T_sd - T_sn)
    F_a = ((C_pa * P_s * t_a)/g) * (T_ad - T_an)

    f_1 = 0.5 * (L_star / (4 * np.pi * (a_plc[i] * a_earth)**2)) * (1 - A) + em_d * sb * T_ad**4 - sb * T_sd**4 - F_o
    f_2 = sb * em_d * T_sd**4 - 2 * sb * em_d * T_ad**4 - F_a
    f_3 = em_n * sb * T_an**4 + F_o - sb * T_sn**4
    f_4 = sb * em_d * T_sn**4 + F_a - 2 * sb * em_n * T_an**4
    return f_1, f_2, f_3, f_4

  for alpha in range(20):
    guess_d = randint(200, 500)
    guess_n = randint(100, 400)
    sol7 = fsolve(f7, [guess_d, guess_d, guess_n, guess_n])
    sol7b[0] = sol7[0] if sol7[0] > 0 and sol7[0] != guess_d else sol7b[0]
    sol7b[1] = sol7[1] if sol7[1] > 0 and sol7[1] != guess_d else sol7b[1]
    sol7b[2] = sol7[2] if sol7[2] > 0 and sol7[2] != guess_n else sol7b[2]
    sol7b[3] = sol7[3] if sol7[3] > 0 and sol7[3] != guess_n else sol7b[3]
  
  print(sol7b)
  sol7b_rounded = ['%.5f' % elem for elem in sol7b]
  
  Tsols.append(sol7b_rounded)
  T_sdlist.append(round(sol7b[0], 5))
  T_adlist.append(round(sol7b[1], 5))
  T_snlist.append(round(sol7b[2], 5))
  T_anlist.append(round(sol7b[3], 5))

Tsols = np.array(Tsols).tolist()
T_snlist = np.array(T_snlist).tolist()
T_sdlist = np.array(T_sdlist).tolist()


print(Tsols)

T_df = pd.DataFrame(np.column_stack([plname2, T_sdlist, T_adlist, T_snlist, T_anlist]), columns = ['Planet name', 'T_sd', 'T_ad', 'T_sn', 'T_an'])
print(T_df)

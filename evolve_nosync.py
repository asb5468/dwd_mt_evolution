import numpy as np
import sys
import os
from scipy.integrate import solve_ivp
from integrate import *

def main(donor_mass, accretor_mass, tau0, basedir):
    method = "DOP853"
    filename = "{}/marsh_{}_{}_{}_{}_Eddington_nosync.dat".format(
        basedir, tau0, donor_mass, accretor_mass, method
    )
    if os.path.exists(filename):
        print("Run already complete, exiting.")
        return

    Md = [donor_mass * Msun]
    Ma = [accretor_mass * Msun]
    sep = [
        rad_zero_temp(Md[-1])
        * Rsun
        / normalized_roche_lobe_egg(donor_mass / accretor_mass)
    ]  # in m
    time1 = [0]
    tau_a = [tau0]
    evol_time = 1e10 - time1[-1]
    norm = tau_a[-1] / (
        (Ma[-1] / Md[-1]) ** 2 * (sep[-1] / (Rsun * rad_zero_temp(Ma[-1]))) ** 6
    )  # in yr
    Omega_o = 2 * np.pi / period(Ma[-1], Md[-1], sep[-1])
    Mdot_o = accretion_rate(Ma[-1], Md[-1], sep[-1])
    Mdot_Edd_o = Eddington_rate_synchronized(Ma[-1], Md[-1], sep[-1]) / Msun * yr_to_sec
    print(Mdot_Edd_o)

    # initial spin from Eq. 35 of Marsh
    Omega_a = [
        3
        * tau0
        * yr_to_sec
        / 2
        * np.sqrt(G * (Md[-1] + Ma[-1]))
        / sep[-1] ** (5.0 / 2)
        * a_dot(Ma[-1], Md[-1], sep[-1], Omega_o, norm)
        + Omega_o
    ]
    if Omega_a[-1] < 0:
        Omega_a = [0]
    print(Omega_a[-1] / Omega_o)

    # Integrate
    initial_conditions = np.array([Ma[-1], Md[-1], sep[-1], Omega_a[-1]])
    t_eval = np.logspace(-2, np.log10(evol_time), 5000) * yr_to_sec
    evol_time *= yr_to_sec
    sol = solve_ivp(
        ode_wrapper,
        [0, evol_time],
        initial_conditions,
        method=method,
        vectorized=True,
        args=(norm,),
    )
    f_gw = 2 * f_orb(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    f_chirp_num = np.gradient(f_gw, sol.t)
    f_chirp = f_dot(sol.y[0], sol.y[1, :], sol.y[2, :], sol.y[3, :], norm)
    fdot_GW = f_dot_GR(sol.y[0], sol.y[1, :], sol.y[2, :])
    fdot_tides = f_dot_tides(sol.y[0], sol.y[1, :], sol.y[2, :], sol.y[3, :], norm)
    fdot_MT = f_dot_MT(sol.y[0], sol.y[1, :], sol.y[2, :])
    Ma_dot = -accretion_rate(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    Omega_o_arr = 2 * np.pi / period(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    Omega_s = sol.y[3, :] / Omega_o_arr

    np.savetxt(
        filename,
        np.array(
            [
                sol.t / yr_to_sec,
                f_chirp * yr_to_sec**2,
                fdot_GW * yr_to_sec**2,
                fdot_MT * yr_to_sec**2,
                fdot_tides * yr_to_sec**2,
                sol.y[0, :] / Msun,
                sol.y[1, :] / Msun,
                Ma_dot * yr_to_sec / Msun,
                Omega_s,
                sol.y[2, :] / Rsun,
            ]
        ).T,
        delimiter="\t",
        header="time (yr)\t fdot_tot (yr^-2)\t fdot_gw\t fdot_MT\t fdot_tides\t Ma (Msun)\t Md (Msun)\t accretion_rate (Msun/yr)\t accretor_spin/orbital_spin\t separation (Rsun)",
    )
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    else:
        print("Usage is")
        print(">python evolve_nosync.py donor_mass accretor_mass tau0 basedir")
        print("donor_mass is the mass of the donor in Msun")
        print("donor_mass is the mass of the accretor in Msun")
        print("tau0 is the initial synchronization timescale in years")
        print("basedir is the directory where the output will be written")

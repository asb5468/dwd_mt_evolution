import numpy as np
import pdb

G = 6.67430e-11  # SI units
h = 6.626e-34  # SI units
c = 299792458  # m/s
Msun = 1.98847e30  # kg
Rsun = 6.957e8  # m
chandra_mass = 1.44 * Msun  # kg
Mp = 0.00057 * Msun  # kg
electron_mass = 9.1e-31  # kg
nucleon_mass = 1.67e-27  # kg
mean_molecular_weight = 2
au_to_rsun = 215
yr_to_sec = 3600 * 24 * 265.25
thomson_cross_section = 6.652587e-29  # m^2


def rad_zero_temp(M):
    # returns units of Rsun
    # Eq. 15 of Verbunt + Rappaport 1988
    return (
        0.0114
        * np.sqrt((M / chandra_mass) ** (-2.0 / 3) - (M / chandra_mass) ** (2.0 / 3))
        * (1 + 3.5 * (M / Mp) ** (-2.0 / 3) + (Mp / M)) ** (-2.0 / 3)
    )


def breakup_rate(M):
    return np.sqrt(G * M / (rad_zero_temp(M) * Rsun) ** 3)


def normalized_roche_lobe_egg(q):
    # Eq. 38 of Kremer+ 2015, q<1
    return 0.49 * q ** (2.0 / 3) / (0.6 * q ** (2.0 / 3) + np.log(1 + q ** (1.0 / 3)))


def rh(q):
    # Eq. 13 of Verbunt + Rappaport 1988
    # q > 1 !!!
    q = 1./q
    return (
        0.0883
        + 0.04858 * np.log10(q)
        + 0.11489 * np.log10(q) ** 2
        - 0.020475 * np.log10(q) ** 3
    )


def rmin(q):
    # Eq. 6 of Nelemans+ 2001, q<1
    return (
        0.04948
        - 0.03815 * np.log10(q)
        + 0.04752 * np.log10(q) ** 2
        - 0.006973 * np.log10(q) ** 3
    )


def moment_of_intertia_constant(accretor_mass):
    # Eq. 23 of Marsh+ 2004, dimensionless
    return 0.1939 * (1.44885 - accretor_mass / Msun) ** 0.1917


def differential_spin(accretor_mass, donor_mass, a, Omega_s):
    Omega_o = 2 * np.pi / period(accretor_mass, donor_mass, a)
    return Omega_s - Omega_o


def period(accretor_mass, donor_mass, a):
    # Kepler's third law
    return np.sqrt(4 * np.pi ** 2 * a ** 3 / (G * (donor_mass + accretor_mass)))


def xL1(q):
    # valid for circular, synchronized donor
    # Eq. A1 of Sepinsk+ y2007, q<1
    return 0.5 + 0.22 * np.log10(q)


def potential_L1(accretor_mass, donor_mass, a):
    # Eq. 14 of Han + Webbink 1999
    q = donor_mass / accretor_mass
    x = xL1(q) * a
    mu = 1 / (1 + q)
    return (
        -G * donor_mass / x
        - G * accretor_mass / (a - x)
        - G * (accretor_mass + donor_mass) / (2 * a ** 3) * (x - mu * a) ** 2
    )


def potential_accretor(accretor_mass, donor_mass, a):
    # Eq. 16 of Han + Webbink 1999
    mu = accretor_mass / (accretor_mass + donor_mass)
    accretor_rad = rad_zero_temp(accretor_mass) * Rsun
    return (
        -G * donor_mass / a
        - G * accretor_mass / accretor_rad
        - G
        * (accretor_mass + donor_mass)
        / (2 * a ** 3)
        * (2.0 / 3 * accretor_rad ** 2 + (a - mu * a) ** 2)
    )


def Eddington_rate_synchronized(accretor_mass, donor_mass, a):
    # Eq. 34 of Marsh+ 2004
    return (
        8
        * np.pi
        * G
        * nucleon_mass
        * c
        * accretor_mass
        / (
            thomson_cross_section
            * (
                potential_L1(accretor_mass, donor_mass, a)
                - potential_accretor(accretor_mass, donor_mass, a)
            )
        )
    )


def zeta_roche_lobe(q):
    # Eq. 18 of Marsh+ 2004
    zeta = (
        (1 + q)
        / 3
        * (2 * np.log(1 + q ** (1.0 / 3)) - (q ** (1.0 / 3)) / (1 + q ** (1.0 / 3)))
        / (0.6 * q ** (2.0 / 3) + np.log(1 + q ** (1.0 / 3)))
    )


def zeta_wd(M):
    return -((M / chandra_mass) ** (-2.0 / 3) + (M / chandra_mass) ** (2.0 / 3)) / (
        3 * ((M / chandra_mass) ** (-2.0 / 3) - (M / chandra_mass) ** (2.0 / 3))
    ) + (14.0 / 3 * (M / Mp) ** (-2.0 / 3) + 2 * (M / Mp) ** (-1)) / (
        3 * (1 + 7.0 / 2 * (M / Mp) ** (-2.0 / 3) + (M / Mp) ** (-1))
    )


def J_orb(accretor_mass, donor_mass, a):
    return np.sqrt(G * a / (donor_mass + accretor_mass)) * donor_mass * accretor_mass


def Jdot_GR(accretor_mass, donor_mass, a):
    return (
        -32.0
        * G ** 3
        * accretor_mass
        * donor_mass
        * (accretor_mass + donor_mass)
        * J_orb(accretor_mass, donor_mass, a)
        / (5 * c ** 5 * a ** 4)
    )


def synchronization_time(accretor_mass, donor_mass, a, norm):
    # Eq. 29 of Marsh+ 2004
    return (
        norm
        * yr_to_sec
        * (accretor_mass / donor_mass) ** 2
        * (a / (rad_zero_temp(accretor_mass) * Rsun)) ** 6
    )


def a_dot_GR(accretor_mass, donor_mass, a):
    return (
        2
        * a
        * Jdot_GR(accretor_mass, donor_mass, a)
        / J_orb(accretor_mass, donor_mass, a)
    )


def a_dot_tides(accretor_mass, donor_mass, a, Omega_s, norm):
    tau = synchronization_time(accretor_mass, donor_mass, a, norm)
    return (
        2
        * a
        * (
            moment_of_intertia_constant(accretor_mass)
            * accretor_mass
            * rad_zero_temp(accretor_mass) ** 2
            * Rsun ** 2
            * differential_spin(accretor_mass, donor_mass, a, Omega_s)
            / (tau * J_orb(accretor_mass, donor_mass, a))
        )
    )


def a_dot_tides_breakup(accretor_mass, donor_mass, a, Omega_s, norm, Omega_sdot):
    tau = synchronization_time(accretor_mass, donor_mass, a, norm)
    omega_over_tau = (
        Omega_sdot
        - Omega_k_dot(accretor_mass, donor_mass, a)
        + differential_spin(accretor_mass, donor_mass, a, Omega_s) / tau
    )
    return (
        2
        * a
        * (
            moment_of_intertia_constant(accretor_mass)
            * accretor_mass
            * rad_zero_temp(accretor_mass) ** 2
            * Rsun ** 2
            * omega_over_tau
            / J_orb(accretor_mass, donor_mass, a)
        )
    )


def a_dot_MT(accretor_mass, donor_mass, a):
    q = donor_mass / accretor_mass
    return (
        -2
        * a
        * (
            (1 - q - np.sqrt((1 + q) * rh(q)))
            * accretion_rate(accretor_mass, donor_mass, a)
            / donor_mass
        )
    )


def a_dot(accretor_mass, donor_mass, a, Omega_s, norm):
    return (
        a_dot_GR(accretor_mass, donor_mass, a)
        + a_dot_tides(accretor_mass, donor_mass, a, Omega_s, norm)
        + a_dot_MT(accretor_mass, donor_mass, a)
    )


def a_dot_breakup(accretor_mass, donor_mass, a, Omega_s, norm, Omega_sdot):
    return (
        a_dot_GR(accretor_mass, donor_mass, a)
        + a_dot_tides_breakup(accretor_mass, donor_mass, a, Omega_s, norm, Omega_sdot)
        + a_dot_MT(accretor_mass, donor_mass, a)
    )


def a_dot_total(accretor_mass, donor_mass, a, Omega_s, norm):
    q = donor_mass / accretor_mass
    tau = synchronization_time(accretor_mass, donor_mass, a, norm)
    return (
        2
        * a
        * (
            Jdot_GR(accretor_mass, donor_mass, a) / J_orb(accretor_mass, donor_mass, a)
            + moment_of_intertia_constant(accretor_mass)
            * accretor_mass
            * rad_zero_temp(accretor_mass) ** 2
            * Rsun ** 2
            * differential_spin(accretor_mass, donor_mass, a, Omega_s)
            / (tau * J_orb(accretor_mass, donor_mass, a))
            - (1 - q - np.sqrt((1 + q) * rh(q)))
            * accretion_rate(accretor_mass, donor_mass, a)
            / donor_mass
        )
    )


def accretion_rate(accretor_mass, donor_mass, a):
    q = donor_mass / accretor_mass
    overfill = Rsun * rad_zero_temp(donor_mass) - normalized_roche_lobe_egg(q) * a
    overfill = np.maximum(overfill, 0)
    mu = donor_mass / (donor_mass + accretor_mass)
    P = period(accretor_mass, donor_mass, a)
    a2 = mu / xL1(q) ** 3 + (1 - mu) / (1 - xL1(q)) ** 3
    f = (
        (8 * np.pi ** 3)
        / 9
        * (5 * G * electron_mass / h ** 2) ** (3.0 / 2)
        * (mean_molecular_weight * nucleon_mass) ** (5.0 / 2)
        * 1.0
        / P
        * (
            3
            * mu
            * donor_mass
            / (5 * normalized_roche_lobe_egg(q) * Rsun * rad_zero_temp(donor_mass))
        )
        ** (3.0 / 2)
        * (a2 * (a2 - 1)) ** (-1.0 / 2)
    )
    return -f * overfill ** 3


def accretion_rate2(accretor_mass, donor_mass, a):
    return 0 * accretor_mass


def Omega_s_dot_disk(accretor_mass, donor_mass, a, Omega_s, norm):
    dlogk_dlogM = - accretor_mass * 0.1917 / (1.44885 * Msun - accretor_mass)
    lambda_k = 1 + 2 * zeta_wd(accretor_mass) + dlogk_dlogM
    q = donor_mass / accretor_mass
    tau = synchronization_time(accretor_mass, donor_mass, a, norm)
    return (
        (
            lambda_k * Omega_s
            - np.sqrt(G * accretor_mass * rad_zero_temp(accretor_mass) * Rsun)
            / (
                moment_of_intertia_constant(accretor_mass)
                * rad_zero_temp(accretor_mass) ** 2
                * Rsun ** 2
            )
        )
        * accretion_rate(accretor_mass, donor_mass, a)
        / (accretor_mass)
        - differential_spin(accretor_mass, donor_mass, a, Omega_s) / tau
    )


def Omega_s_dot(accretor_mass, donor_mass, a, Omega_s, norm):
    dlogk_dlogM = - accretor_mass * 0.1917 / (1.44885 * Msun - accretor_mass)
    lambda_k = 1 + 2 * zeta_wd(accretor_mass) + dlogk_dlogM
    q = donor_mass / accretor_mass
    tau = synchronization_time(accretor_mass, donor_mass, a, norm)
    return (
        (
            lambda_k * Omega_s
            - np.sqrt(G * accretor_mass * rh(q) * a)
            / (
                moment_of_intertia_constant(accretor_mass)
                * rad_zero_temp(accretor_mass) ** 2
                * Rsun ** 2
            )
        )
        * accretion_rate(accretor_mass, donor_mass, a)
        / (accretor_mass)
        - differential_spin(accretor_mass, donor_mass, a, Omega_s) / tau
    )


def Omega_k_dot(accretor_mass, donor_mass, a):
    return (
        -breakup_rate(accretor_mass)
        / 2
        * (1 - 3 * zeta_wd(accretor_mass))
        * accretion_rate(accretor_mass, donor_mass, a)
        / accretor_mass
    )


def f_orb(accretor_mass, donor_mass, a):
    return np.sqrt(G * (accretor_mass + donor_mass) / (4 * np.pi ** 2 * a ** 3))


def f_dot(accretor_mass, donor_mass, a, Omega_s, norm):
    # This is the GW frequency derivative
    a_dot_t = a_dot(accretor_mass, donor_mass, a, Omega_s, norm)
    return (
        -3.0
        / 2
        * np.sqrt(G * (accretor_mass + donor_mass) / (np.pi ** 2 * a ** 5))
        * a_dot_t
    )


def f_dot_GR(accretor_mass, donor_mass, a):
    a_dot_t = a_dot_GR(accretor_mass, donor_mass, a)
    return (
        -3.0
        / 2
        * np.sqrt(G * (accretor_mass + donor_mass) / (np.pi ** 2 * a ** 5))
        * a_dot_t
    )


def f_dot_tides(accretor_mass, donor_mass, a, Omega_s, norm):
    a_dot_t = a_dot_tides(accretor_mass, donor_mass, a, Omega_s, norm)
    return (
        -3.0
        / 2
        * np.sqrt(G * (accretor_mass + donor_mass) / (np.pi ** 2 * a ** 5))
        * a_dot_t
    )


def f_dot_MT(accretor_mass, donor_mass, a):
    a_dot_t = a_dot_MT(accretor_mass, donor_mass, a)
    return (
        -3.0
        / 2
        * np.sqrt(G * (accretor_mass + donor_mass) / (np.pi ** 2 * a ** 5))
        * a_dot_t
    )


def ode_wrapper(t, y, norm):
    # unpack
    accretor_mass = y[0]
    donor_mass = y[1]
    a = y[2]
    Omega_s = y[3]

    # above chandrasekhar mass
    if accretor_mass > chandra_mass:
        print("Exceeds Chandrasekhar mass!")
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # below minimum mass
    donor_rad = rad_zero_temp(donor_mass)
    accretor_rad = rad_zero_temp(accretor_mass)
    total_rad = donor_rad + accretor_rad
    if (donor_mass < 0.01 * Msun) or (a < total_rad):
        print("Donor too small")
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # check numerical issues
    if (accretor_mass < 0) and (donor_mass < 0):
        print("Numerical issue - negative masses!")
    elif accretor_mass < 0:
        print("Numerical issue - negative accretor mass!")
    elif donor_mass < 0:
        print("Numerical issue - negative donor mass!")
    
    # accretion rate
    Md_dot = accretion_rate(accretor_mass, donor_mass, a)
    Mdot_Edd = Eddington_rate_synchronized(accretor_mass, donor_mass, a)
    if Md_dot > 0:
        print("Donor is accreting?")
    if abs(Md_dot) >= abs(Mdot_Edd):
        Ma_dot = Mdot_Edd
    else:
        Ma_dot = -Md_dot

    # unstable mass transfer
    if (abs(Md_dot) / Msun * yr_to_sec) > 0.01:
        print("Unstable")
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # check disk
    rad1 = rad_zero_temp(accretor_mass) * Rsun
    rad_min = rmin(donor_mass / accretor_mass) * a
    if rad1 < rad_min:
        Omega_sdot = Omega_s_dot_disk(accretor_mass, donor_mass, a, Omega_s, norm)
    else:
        Omega_sdot = Omega_s_dot(accretor_mass, donor_mass, a, Omega_s, norm)

    # check breakup rate
    Omega_k = breakup_rate(accretor_mass)
    Omega_kdot = Omega_k_dot(accretor_mass, donor_mass, a)
    if (Omega_s >= Omega_k) and (Omega_sdot > Omega_kdot):
        adot = a_dot_breakup(accretor_mass, donor_mass, a, Omega_s, norm, Omega_sdot)
        Omega_sdot = Omega_kdot
    else:
        adot = a_dot(accretor_mass, donor_mass, a, Omega_s, norm)

    return np.array([Ma_dot, Md_dot, adot, Omega_sdot])

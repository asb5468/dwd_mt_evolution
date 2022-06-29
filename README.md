This repo contains code for evolving double white dwarf systems according to the prescription in [Marsh et. al 2004](https://academic.oup.com/mnras/article/350/1/113/986306) as implemented in Biscoveanu, Kremer, and Thrane 2022.
The script `integrate.py` contains all the functions governing the evolution of the system, like tides, mass transfer, and gravitational radiation. This can be run using `evolve_nosync.py` as follows:
```
python evolve_nosync.py donor_mass accretor_mass tau0 basedir 
```
where `donor_mass` and `accretor_mass` are in solar masses. `tau0` is the initial synchronization time at the onset of mass transfer in years, and `basedir` is the directory where the output will be written. As an example:
```
python evolve_nosync 0.25 0.6 10 ./
```
This will produce an output file called `{basedir}/marsh_{tau0}_{donor_mass}_{accretor_mass}_DOP853.dat`, where `DOP853` is the integration method used by `scipy.solve_ivp`. The output file containts arrays for the following:
- `time (yr)`
- `fdot_tot (yr^-2)`: the total **gravitational-wave** frequency derivative 
- `fdot_gw`: the contribution to the **gravitational-wave** frequency derivative from gravitational radiation, in yr^-2
- `fdot_MT`: the contribution to the **gravitational-wave** frequency derivative from mass transfer, in yr^-2 
- `fdot_tides`: the contribution to the **gravitational-wave** frequency derivative from tides along, in yr^-2
- `Ma (Msun)`: the accretor mass
- `Md (Msun)`: the donor mass
- `accretion_rate (Msun/yr)`
- `accretor_spin/orbital_spin`: the accretor spin in units of the orbital angular frequency
- `separation (Rsun)`

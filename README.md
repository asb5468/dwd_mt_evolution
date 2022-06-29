This repo contains code for evolving double white dwarf systems according to the prescription in [Marsh et. al 2004](https://academic.oup.com/mnras/article/350/1/113/986306) as implemented in Biscoveanu, Kremer, and Thrane 2022.
The script `integrate.py` contains all the function governing the evolution of the system, like tides, mass transfer, and gravitational radiation. This can be run using `evolve_nosync.py` as follows:
```
python evolve_nosync.py donor_mass accretor_mass tau0 basedir 
```
where `donor_mass` and `accretor_mass` are in solar masses. `tau0` is the initial synchronization time at the onset of mass transfer is in years, and `basedir` is the directory where the output will be written. As an example:
```
python evolve_nosync 0.25 0.6 10 ./
```
This will produce an output file called `{basedir}/marsh_{tau0}_{donor_mass}_{accretor_mass}_DOP853.dat`, where `DOP853` is the integration method used by `scipy.solve_ivp`. The output file containts arrays for the time, the contributions to the **gravitational-wave** frequency derivative from gravitational radiation, mass transfer, and tides along with the total gravitational-wave frequency derivative, the accretor and donor masses, the accretion rate, the accretor spin, and the separation.

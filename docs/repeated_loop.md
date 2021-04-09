# Repeated Loop Car-Racing Simulator
The repeated loop racing simulator serves as a toolkit for testing different algorithm for car racing in a user-designed track.

#### Instructions

##### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `car-racing`, and you could also choose that name with your own preferences by editing the .yml file.

##### Examples
###### System identifications
```
python scripts/repeated_loop/identification.py
``` 
This allows to identify the linearized dynamics of the racing car by regression.
###### Tracking 
```
python scripts/repeated_loop/tracking.py --ctrl-policy mpc-lti --simulation --plotting --animation
```
This allows to test algorithm for tracking. The choices of `--ctrl-policy` could be `mpc-lti` and `pid`. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.

###### Racing
```
python scripts/repeated_loop/racing.py --simulation --plotting --animation
```
This allows to test algorithm for racing. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.
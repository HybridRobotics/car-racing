#  Offboard Car-Racing Simulator
The offboard racing simulator serves as a toolkit for testing different algorithm for car racing in a user-designed track.

#### Instructions

##### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `car-racing`, and you could also choose that name with your own preferences by editing the .yml file.
* Currently the repository is not created as standard package through setuptools. In order to run examples, run following command in terminal under root folder `car_racing`.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

##### Examples
###### System identifications
```
python scripts/offboard/identification.py
``` 
This allows to identify the linearized dynamics of the racing car by regression.
###### Tracking 
```
python scripts/offboard/tracking.py --ctrl-policy mpc-lti --track-layout l_shape --simulation --plotting --animation 
```
This allows to test algorithm for tracking. The choices of `--ctrl-policy` could be `mpc-lti` and `pid`. `--track-layout` could be `l_shape`, `goggle` and `ellipse`. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.

###### Racing
```
python scripts/offboard/racing.py --track-layout l_shape --simulation --plotting --animation
```
This allows to test algorithm for racing. `--track-layout` could be `l_shape`, `goggle` and `ellipse`. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `--animation` are currently purely optional.

###### LMPC Racing
```
python scripts/offboard/lmpc_racing.py --track-layout l_shape --lap-number 8 --simulation --plotting --animation
```
This allows to test algorithm for learning based mpc. `--lap-number` indicates the total numbers of the lap. `--track-layout` could be `l_shape`, `goggle` and `ellipse`. `--simulation`, `--track-layout` and `--lap-number` are no longer required once you have generated .obj file for simulation data. `--plotting`, `--animation` are currently purely optional. `--plotting` optition will show you the trajectory, state and input of the ego vehicle. `--animation` will save the animation of the ego vehicle's lap.

###### Racing Game

```
python scripts/offboard/racing_game.py --track-layout l_shape --lap-number 10 --simulation --direct-lmpc --animation --sim-back
```
This allows to test algorightm for racing environment with moving obstacles. In addition to the above options, following options are available:
`--direct-lmpc`: This allows the simulator load stored trajectories and start LMPC algorithm directly.
`--diff-alpha`: This allows the simulator do 10 simulations with different alpha value (weight for reference path in the planner).
`--random-sim`: This allows the simulator do 10 simulations with moving obstacles start from different position and speed.
`--diy-sim`: This allows the simulator do the simulation with self-defined scenario.
`--sim-back`: By changing the file path, this allows the simulator do the simulation with differnt parameters but with the same scenario.

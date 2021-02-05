**Status:** This repository is still under development, expecting new features/papers and a complete tutorial to explain it. Feel free to raise questions/suggestions through GitHub Issues, if you want to use the current version of this repository.

# Car-Racing
This repo serves as a toolkit for testing different algorithm for car racing in a user-designed track.

#### References
The code in this repository is based on the following:
* J. Zeng, B. Zhang and K. Sreenath. "Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function." 2021 American Control Conference (ACC). [PDF](https://arxiv.org/pdf/2007.11718.pdf) 

#### Instructions

##### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `car-racing`, and you could also choose that name with your own preferences by editing the .yml file.

##### Examples
###### System identifications
```
python examples/identification.py
``` 
This allows to identify the linearized dynamics of the racing car by regression.
###### Tracking 
```
python examples/tracking.py --ctrl-policy mpc-lti --simulation --plotting --animation
```
This allows to test algorithm for tracking. The choices of `--ctrl-policy` could be `mpc-lti` and `pid`. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.

###### Racing
```
python examples/racing.py --simulation --plotting --animation
```
This allows to test algorithm for racing. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.

#### Acknowledgements
I would like to say great thanks to [Ugo Rosolia](https://github.com/urosolia), as some helper functions in this repository come from his open source repo [RacingLMPC](https://github.com/urosolia/RacingLMPC).
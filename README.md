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
* Run `python examples/identification.py` to run system identification over single racing car.
* Run `python examples/tracking.py` to run PID or MPC tracking the centerline of the track.
* Run `python examples/racing.py` to run racing simulation between ego car and surrounding cars.

#### Acknowledgements
I would like to say great thanks to [Ugo Rosolia](https://github.com/urosolia), as some helper functions in this repository come from his open source repo [RacingLMPC](https://github.com/urosolia/RacingLMPC).
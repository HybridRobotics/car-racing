**Status:** Under development (expect new features and new tutorial)

# Car-Racing
This repo serves as a toolkit for testing different algorithm for car racing in a user-designed track.

#### References
The code in this repository is based on the following:
* J. Zeng, B. Zhang and K. Sreenath. "Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function." 2021 American Control Conference (ACC). [PDF](https://arxiv.org/pdf/2007.11718.pdf) 

#### Instructions
##### Environments
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `car-racing`, and you could also choose that name with your own preferences by editing the .yml file.

##### Examples
- Run `examples/identification.py` to run system identification over single racing car.
- Run `examples/tracking.py` to run PID or MPC tracking the centerline of the track.
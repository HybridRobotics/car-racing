**Status:** This repository is still under development, expecting new features/papers and a complete tutorial to explain it. Feel free to raise questions/suggestions through GitHub Issues, if you want to use the current version of this repository.

Car Racing
==========

This repository provides a toolkit to test control and planning problems for car racing simulation environment.

<details>
 <summary>Click to open Table of Contents</summary>

## Table of Contents
- [References](#references)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
    - [Offboard](#offboard)
        - [System Identification](#system-identification)
        - [Tracking](#tracking)
        - [Racing](#racing)
    - [Realtime](#realtime)
- [Author](#author)
- [Contributing](#contributing)
</details>

## References
If you find this project useful in your work, please consider citing following papers:

* J. Zeng, B. Zhang and K. Sreenath. "Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function." 2021 American Control Conference (ACC). [[PDF]](https://arxiv.org/pdf/2007.11718.pdf)

## Features

## Installation
* Create your environment via `conda env create -f environment.yml`. The default conda environment name is `car-racing`, and you could also choose that name with your own preferences by editing the .yml file.
* Currently the repository is not created as standard package through setuptools. In order to run examples, run following command in terminal to add root folder into your `PYTHONPATH`.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Usage

### Offboard
#### System Identification
Run
```
python scripts/offboard/identification.py
``` 
This allows to identify the linearized dynamics of the racing car by regression.

#### Tracking
Run
```
python scripts/offboard/tracking.py --ctrl-policy mpc-lti --track-layout l_shape --simulation --plotting --animation 
```
This allows to test algorithm for tracking. The choices of `--ctrl-policy` could be `mpc-lti` and `pid`. `--track-layout` could be `l_shape`, `m_shape`, `goggle` and `ellipse`. `--simulation` is no longer required once you have generated .obj file for simulation data. `--plotting`, `animation` are currently purely optional.

#### Racing

### Realtime
To start the simulator, run the following command in terminal:
```
roslaunch car_racing car_racing_sim.launch track_layout:=goggle
```
This allows you to run the simulator and visualization node. Change the `track_layout`, you can get differnt tracks. The center line of the race track is plotted in red dash line; the optimal trajectory of the race track is plotted in green line.

To add new vehicle with controller in the simulator, run the following commands in new terminals:
```
rosrun car_racing vehicle.py --veh-name vehicle1 --color blue --vx 0 --vy 0 --wz 0 --epsi 0 --s 0 --ey 0

rosrun car_racing controller.py --ctrl-policy mpc-lti --veh-name vehicle1
```
Where `--veh-name` is a self-defined vehicle's name, `--color` is the color of the vehicle in the animation. `--vs, --vy, --wz, --epsi, --s, --ey` is the initial state of the vehicle in Frenet coordinate. The choices of ` --ctrl-policy` could be `pid`, `mpc-lti` and `mpc-cbf`. 

## Author
- [Jun Zeng](https://github.com/junzengx14)
- [Suiyi He](https://github.com/hesuieins)

## Contributing
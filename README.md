**Status:** This repository is still under development, expecting new features/papers and a complete tutorial to explain it. Feel free to raise questions/suggestions through GitHub Issues, if you want to use the current version of this repository.

car-racing
==========

This repository provides a toolkit to test control and planning problems for car racing simulation environment.
<details>
 <summary>Click to open Table of Contents</summary>

## Table of Contents
- [Preview](#preview)
- [References](#references)
- [Features](#features)
- [Installation](#installation)
- [Quick-Demos](#quick-demos)
- [Docs](#docs)
    - [Offboard](#offboard)
- [Appendix](#appendix)
</details>

## Preview
<img src="/media/release/icra2022.gif?raw=true" width="800"/>

## References
This project was developed by the help from the following papers:

Parallelized optimization for overtake racing behavior with multiple autonomous vehicles [[IEEE]](https://ieeexplore.ieee.org/document/9811969) [[arXiv]](https://arxiv.org/abs/2112.06435) [[Video]](https://youtu.be/1zTXfzdQ8w4)
```
@inproceedings{he2022parallel,
  title={Autonomous racing with multiple vehicles using a parallelized optimization with safety guarantee using control barrier functions},
  author={He, Suiyi and Zeng, Jun and Sreenath, Koushil},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

Design model predictive control with control barrier functions for obstacle avoidance in car racing problems [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9483029) [[arXiv]](https://arxiv.org/abs/2007.11718) [[NorCal Control Workshop Talk]](https://youtu.be/IfNgogcLSjE)
```
@inproceedings{zeng2021mpccbf,
  title={Safety-critical model predictive control with discrete-time control barrier function},
  author={Zeng, Jun and Zhang, Bike and Sreenath, Koushil},
  booktitle={2021 American Control Conference (ACC)},
  year={2021},
  volume={},
  number={},
  pages={3882-3889}
}
```

## Features

## Installation
* We recommend creating a new conda environment:
```
conda env create -f environment.yml
conda activate car-racing
```

Run following command in terminal to install the car racing simulator package.
```
pip install -e .
```

## Auto Testing

In this project, `pytest` is used to test the code autonomously after pushing new code to the repository. Currently, three files in the `tests` folder are used for testing pid or mpc tracking controller, mpc-cbf controller and racing game planner, respectively. To test other features, add files to the `tests` folder and update the `tests.yml` file under the `.github/workflows` folder.



## Quick-Demos

## Docs
The following documentation contains documentation and common terminal commands for simulations and testing.

### Offboard


#### Racing competition with ego controller (LMPC)
To save the historic states and inputs used for learning-based MPC, run the following command for each track layout firstly:
```
python car_racing/tests/lmpc_test.py \
--track-layout l_shape --lap-number 7 --simulation --save-trajectory
```
Then you can run the following command: 
```
python car_racing/tests/lmpc_test.py \
--track-layout l_shape --lap-number 10 --simulation --direct-lmpc --animation --plotting
```
This allows to test algorithm for learning-based MPC. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `track_layout` | string | `l_shape`, `m_shape`, `goggle`, `ellipse` | track layouts |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `direct_lmpc` | action | `store_true` | if true, the simulator will begin the LMPC controller directly using store trajectories |
| `zero_noise` | action | `store_true` | no noises in dynamic update if true |
|`save_trajectory`| action |`store_true`|if true and when the controller is LMPC, simulator will store the history states and inputs|
| `simulation` | action | `store_true` | generate simulation data if true, otherwise read simulation data from existing files |
| `plotting` | action | `store_true` | save plotting if true |
| `animation` | action | `store_true` | save animation if true |

#### Racing competition with ego planner and controller
To save the historic states and inputs used for learning-based MPC, run the following command for each track layout firstly:
```
python car_racing/tests/overtake_planner_test.py \
--track-layout l_shape --lap-number 7 --simulation --number-other-agents 0 --save-trajectory
```
Then you can run the following command: 
```
python car_racing/tests/overtake_planner_test.py \
--track-layout l_shape --lap-number 10 --simulation --direct-lmpc --animation --plotting --number-other-agents 3
```
This allows to test algorithm for racing competition. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `track_layout` | string | `l_shape`, `m_shape`, `goggle`, `ellipse` | track layouts |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `direct_lmpc` | action | `store_true` | if true, the simulator will begin the LMPC controller directly using store trajectories |
| `sim_replay` | action | `store_true` | if true, by changingfile path, the simulator will simulate with different parameters but from same initial conditions |
| `zero_noise` | action | `store_true` | no noises in dynamic update if true |
| `diff_alpha` | action | `store_true` | if true, different alpha values will be used for same initial conditions |
| `random_other_agents` | action | `store_true` | other agents will be generated randomly if true |
| `number_other_agents` | int | any number that is greater than `0`, when it is set to `0`, the algorithm is LMPC | number of agents that will be generated |
|`save_trajectory`| action |`store_true`|if true and when the controller is LMPC, simulator will store the history states and inputs|
|`multi_tests`| action | `store_true`| if ture, 100 groups of randomly generated tests will be simulated|
| `simulation` | action | `store_true` | generate simulation data if true, otherwise read simulation data from existing files |
| `plotting` | action | `store_true` | save plotting if true |
| `animation` | action | `store_true` | save animation if true |


Currently, path planner and trajecotry planner are available for the overtaking maneuver. Changing the varibale `self.path_planner` in `base.py` to `True` allows the controller to simulate with path planner.

### Other testing mode (under development)
The test modes below are from the old project of the paper we referenced. Since it is not nessesary for our project, we haven't refactor the code for it to run on the new version of this codebase. Needs some refactoring job before using the test below.

#### System Identification
Run
```
python car_racing/tests/system_identification_test.py
```
This allows to identify the linearized dynamics of the racing car by regression.

#### Tracking performance with controllers
Run
```
python car_racing/tests/control_test.py --ctrl-policy mpc-lti --track-layout l_shape --simulation --plotting --animation 
```
This allows to test algorithm for tracking. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `ctrl_policy` | string | `pid`, `mpc-lti`, `lqr` | control policy |
| `track_layout` | string | `l_shape`, `m_shape`, `goggle`, `ellipse` | track layouts |
| `simulation` | action | `store_true` | generate simulation data if true, otherwise read simulation data from existing files |
| `plotting` | action | `store_true` | save plotting if true |
| `animation` | action | `store_true` | save animation if true |

#### Racing competition with ego controller (MPC-CBF)
Run
```
python car_racing/tests/mpccbf_test.py --track-layout l_shape --simulation --plotting --animation
```
This allows to test algorithm for MPC-CBF controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `track_layout` | string | `l_shape`, `m_shape`, `goggle`, `ellipse` | track layouts |
| `simulation` | action | `store_true` | generate simulation data if true, otherwise read simulation data from existing files |
| `plotting` | action | `store_true` | save plotting if true |
| `animation` | action | `store_true` | save animation if true |

#### Racing competition with ego controller (iLQR)
Run

```
python car_racing/tests/ilqr_test.py --track-layout l_shape --simulation --plotting --animation
```

This allows to test algorithm for iLQR controller. The argparse arguments are listed as follow,

|      name      |  type  |                  choices                  |                         description                          |
| :------------: | :----: | :---------------------------------------: | :----------------------------------------------------------: |
| `track_layout` | string | `l_shape`, `m_shape`, `goggle`, `ellipse` |                        track layouts                         |
|  `simulation`  | action |               `store_true`                | generate simulation data if true, otherwise read simulation data from existing files |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |
|  `animation`   | action |               `store_true`                |                    save animation if true                    |

## Appendix
from detail explaination of our project please refer to our documentations below:

 Project report: https://drive.google.com/file/d/1qqdYlhszkwPwZETbUJ6PtY3SAp_if8y6/view?usp=sharing
 
 project presentation: https://docs.google.com/presentation/d/1UmvFBaMAFYTbx05MaQgZgqfkojcb9ZeR/edit?usp=sharing&ouid=106817078880653835914&rtpof=true&sd=true
 
 Proejct website: https://eecs106b-banana-radiation.github.io/

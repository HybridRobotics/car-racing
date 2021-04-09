# Real-Time Car-Racing Simulator
The real-time racing simulator serves as a real-time toolkit for testing different algorithm for car racing in a user-designed track.

#### Instructions

##### Environments
Currently, the repository is tested with ROS Melodic on a Ubuntu 18.04 laptop. The conda environment is not used for this online simulator.


##### Examples
###### Start Simulation
```
roslaunch car_racing car_racing_sim.launch ctrl_policy:=pid
```
This allows you to run the simulator. The choices of ` ctrl_policy` could be `pid` and `mpc-lti`. Currently, only tracking function is defined in this simulator.
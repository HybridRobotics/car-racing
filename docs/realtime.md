# Real-Time Car-Racing Simulator
The real-time racing simulator serves as a real-time toolkit for testing different algorithm for car racing in a user-designed track.

#### Instructions

##### Environments
Currently, the repository is tested with ROS Melodic on a Ubuntu 18.04 laptop. The conda environment is not used for this online simulator.


##### Examples
###### Start Simulation
Currently the repository is not created as standard package through setuptools. In order to run the simulator, run following command in all used terminals under root folder `car_racing`.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```


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
import pickle
import numpy as np
from racing import offboard
from utils import base, racing_env
from utils.constants import *

def test_tracking():
    track_spec = np.genfromtxt("data/track_layout/l_shape.csv", delimiter=",")
    track = racing_env.ClosedTrack(track_spec, track_width=0.8)
    # setup ego car
    ego = offboard.DynamicBicycleModel(name="ego", param=base.CarParam(edgecolor="black"), system_param = base.SystemParam())
    ego.set_zero_noise()
    ego.set_state_curvilinear(np.zeros((X_DIM,)))
    ego.set_state_global(np.zeros((X_DIM,)))
    ego.start_logging()
    ego.set_ctrl_policy(offboard.PIDTracking(vt=0.8))
    ego.ctrl_policy.set_timestep(0.1)
    ego.ctrl_policy.set_track(track)
    ego.set_track(track)
    # setup simulation
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    ego.ctrl_policy.set_racing_sim(simulator)
    simulator.sim(sim_time=20.0)
    mpc_lti_param = base.MPCTrackingParam(vt=0.8)
    ego.set_ctrl_policy(offboard.MPCTracking(mpc_lti_param, ego.system_param))
    ego.ctrl_policy.set_timestep(0.1)
    ego.ctrl_policy.set_track(track)
    ego.ctrl_policy.set_racing_sim(simulator)
    simulator.sim(sim_time=20.0)
    with open("data/simulator/tracking.obj", "wb") as handle:
        pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    simulator.plot_simulation()
    simulator.plot_state("ego")
    simulator.animate(filename="tracking", ani_time=40, imagemagick=True)
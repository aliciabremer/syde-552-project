import numpy as np
import nengo
import matplotlib.pyplot as plt
import sys, os
# set path
from sspslam.sspspace import HexagonalSSPSpace
import argparse
from models.pathintegration import get_path_integrator, get_path_integrator_triple, get_path_integrator_phase, get_path_integrator_shift 
import time

# General process of running path integrator on random path similar to 
# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/new/experiments/run_pathint.py

time = 10 
dt = 0.001
timesteps = np.arange(0, time, dt)

domain_dim = 2 # args.domain_dim
ssp_dim = 55 # args.ssp_dim
length_scale = 0.1 # args.length_scale
radius = 1

# will need to change seed for each

path = np.hstack([
    nengo.processes.WhiteSignal(time, high=0.8, seed=1+i).run(time,dt=dt) for i in range(domain_dim)
    # nengo.processes.WhiteSignal(time, high=args.limit, seed=args.seed+i) for i in range(domain_dim)
])
pathlen = path.shape[0]

vels = (1/dt) * (path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int), :] -
    path[(np.minimum(np.floor(timesteps/dt), pathlen-2).astype(int)), :])


# SSP_SPACE
bounds = np.vstack([np.min(path,axis=0)*1.5, np.max(path,axis=0)*1.5]).T
ssp_space = HexagonalSSPSpace(
    domain_dim = domain_dim,
    ssp_dim = ssp_dim,
    domain_bounds = bounds,
    length_scale = length_scale,
    # rng = 0 # args.seed
)
d = ssp_space.ssp_dim
real_ssp = ssp_space.encode(path)

scale_fac = 1/np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled = vels * scale_fac

pi_n_neurons = 500 # args.pi_n_neurons
gc_n_neurons = 1000 # args.number_gcs
tau = 0.05
n_timesteps = timesteps.shape[0]

model = nengo.Network(seed=1)
with model:
    vel_input = nengo.Node(lambda t : vels_scaled[int(np.minimum(np.floor(t/dt), n_timesteps-1))])
    init_state = nengo.Node(lambda t : real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))]  if t<0.05 else np.zeros(d) )

    if False:
        pathintegrator = get_path_integrator_shift(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True,
                  error_correction_factor=0.2)
    elif False:
        pathintegrator = get_path_integrator_triple(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True,
                  error_correction_factor=0.2)
    elif False:
        pathintegrator = get_path_integrator_phase(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True,
                  error_correction_factor=0.2)
    else:
        pathintegrator = get_path_integrator(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True)
        
    nengo.Connection(vel_input, pathintegrator.velocity_input, synapse=None)
    nengo.Connection(init_state, pathintegrator.input, synapse=None)

    ssp_probe = nengo.Probe(pathintegrator.output, synapse=0.05)

import numpy as np
import nengo
import matplotlib.pyplot as plt
import sys, os
# set path
from sspslam.sspspace import HexagonalSSPSpace
import argparse
from models.pathintegration import get_path_integrator, get_path_integrator_triple, get_path_integrator_phase, get_path_integrator_shift
import time
import scipy
import sklearn.metrics
from utils import sparsity_to_x_intercept

# General process of running path integrator on random path similar to 
# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/new/experiments/run_pathint.py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--domain_dim", default=2, type=int)
parser.add_argument("--limit", default=0.08, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--time_sec", default=10, type=float)
parser.add_argument("--pi_n_neurons", default=700, type=int)
parser.add_argument("--ssp_dim", default=55, type=int)
parser.add_argument("--grid_cells", default=True, type=bool)
parser.add_argument("--number_gcs", default=1000, type=int)
parser.add_argument("--length_scale", default=0.1, type=float)
parser.add_argument("--error_corrector", default=0.1, type=float)

args = parser.parse_args()

time_sec = args.time_sec
dt = 0.001
timesteps = np.arange(0, time_sec, dt)

domain_dim = args.domain_dim
ssp_dim = args.ssp_dim
length_scale = args.length_scale
radius = 1

# will need to change seed for each

path = np.hstack([
    nengo.processes.WhiteSignal(time_sec, high=args.limit, seed=1+i).run(time_sec,dt=dt) for i in range(domain_dim)
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
    length_scale = length_scale
)
d = ssp_space.ssp_dim
real_ssp = ssp_space.encode(path)

scale_fac = 1/np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled = vels * scale_fac

pi_n_neurons = args.pi_n_neurons
gc_n_neurons = args.number_gcs
tau = 0.05
n_timesteps = timesteps.shape[0]

coupling_methods = ['basic', 'triple', 'phase', 'shift']

# fix encoders and intersects
encoders = ssp_space.sample_grid_encoders(gc_n_neurons)
intersects = nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)])

for c in coupling_methods:
    model = nengo.Network(seed=1)
    with model:
        vel_input = nengo.Node(lambda t : vels_scaled[int(np.minimum(np.floor(t/dt), n_timesteps-1))])
        # vel_input = nengo.Node(lambda t : [0,0])
        init_state = nengo.Node(lambda t : real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))]  if t<0.05 else np.zeros(d) )


        if c == "basic":
            print('basic')
            pathintegrator = get_path_integrator(ssp_space, pi_n_neurons, tau, 
                    scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True,
                    intercepts=intersects, encoders=encoders)
        elif c == "triple":
            print("triple")
            pathintegrator = get_path_integrator_triple(ssp_space, pi_n_neurons, tau, 
                    scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                    n_correctors=args.n_corrector, error_correction_factor=args.error_corrector, stable=True,
                    intercepts=intersects, encoders=encoders)
        elif c == "phase":
            print("phase")
            pathintegrator = get_path_integrator_phase(ssp_space, pi_n_neurons, tau, 
                    scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                    n_correctors=args.n_corrector, error_correction_factor=args.error_corrector, stable=True,
                    intercepts=intersects, encoders=encoders)
        elif c == "shift":
            print("shift")
            pathintegrator = get_path_integrator_shift(ssp_space, pi_n_neurons, tau, 
                    scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                    error_correction_factor=args.error_corrector, stable=True,
                    intercepts=intersects, encoders=encoders)
        else:
            pathintegrator = get_path_integrator(ssp_space, pi_n_neurons, tau, 
                    scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True,
                    intercepts=intersects, encoders=encoders)
            
        nengo.Connection(vel_input, pathintegrator.velocity_input, synapse=None)
        nengo.Connection(init_state, pathintegrator.input, synapse=None)

        ssp_probe = nengo.Probe(pathintegrator.output, synapse=0.05)
        print(pathintegrator.output.neurons.probeable)
        print(pathintegrator.output.probeable)
        spike_probe = nengo.Probe(pathintegrator.neurons, synapse=0.05)
            
        
    
    with nengo.Simulator(model) as sim:
        sim.run(args.time_sec)

    sim_path_est = ssp_space.decode(sim.data[ssp_probe], 'from-set','grid', 100)

    filename = c + '_sspdim_' + str(d) + '_pinneurons_' + str(pi_n_neurons) + '_T_' + str(int(time_sec)) + '_seed_' + str(args.seed) + '_time_' + str(time.time()) + '.npz'
    np.savez("data_grid/" + filename, ts = sim.trange(),path=path,real_ssp=real_ssp,
              pi_sim_out = sim.data[ssp_probe],
              pi_sims = np.sum(sim.data[ssp_probe]*real_ssp,axis=1)/np.linalg.norm(sim.data[ssp_probe],axis=1),
              pi_path = sim_path_est, 
              pi_error =np.sqrt(np.sum((path - sim_path_est)**2,axis=1)),
              neurons_out = sim.data[spike_probe])

def exact_path_integration(vels, init, dt):
    output_path = np.zeros(vels)
    output_path[0] = init
    for i in range(1, vels.shape[0]):
        # solve differential equation with trapezoidal method
        output_path[i] = output_path[i-1] + dt/2 * (vels[i-1] + vels[i])
    return output_path

output_path = exact_path_integration(vels_scaled, path[0], dt)


model = nengo.Network(seed=1)
with model:
    output_node = nengo.Node(lambda t : output_path[np.floor(t/dt).astype(int)] )
    output_grid = nengo.Ensemble(
        gc_n_neurons,d,encoders=encoders,
        intercepts=intersects, # idk why this
        label='output grid')
        
    nengo.Connection(output_node, output_grid, 
                            # transform = ssp_space.encode,
                            synapse=0.05)
        
    spike_probe = nengo.Probe(output_grid.output.neurons, synapse=0.05)
    
        
with nengo.Simulator(model) as sim:
    sim.run(args.time_sec)

# sim_path_est = ssp_space.decode(sim.data[ssp_probe], 'from-set','grid', 100)

filename = 'exact_sspdim_' + str(d) + '_pinneurons_' + str(pi_n_neurons) + '_T_' + str(int(time_sec)) + '_seed_' + str(args.seed) + '_time_' + time.time() + '.npz'
np.savez("data_grid/" + filename, ts = sim.trange(),path=path,real_ssp=real_ssp,
              # pi_sim_out = sim.data[ssp_probe],
              #pi_sims = np.sum(sim.data[ssp_probe]*real_ssp,axis=1)/np.linalg.norm(sim.data[ssp_probe],axis=1),
              # pi_path = sim_path_est, 
              #pi_error =np.sqrt(np.sum((path - sim_path_est)**2,axis=1)),
            output_path=output_path,
            neurons_out = sim.data[spike_probe])

plt.figure(dpi=200)
plt.title("spiking")
plt.plot(sim.trange(), sim.data[spike_probe][0])
plt.show()




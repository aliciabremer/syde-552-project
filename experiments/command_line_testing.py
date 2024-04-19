import numpy as np
import nengo
import matplotlib.pyplot as plt
import sys, os
# set path
# from ssp import HexSSP
from sspslam.sspspace import HexagonalSSPSpace
import argparse
from models.pathintegration import get_path_integrator, get_path_integrator_triple, get_path_integrator_phase, get_path_integrator_shift
import time
import scipy
import sklearn.metrics

# General process of running path integrator on random path similar to 
# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/new/experiments/run_pathint.py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--domain_dim", default=2, type=int)
parser.add_argument("--limit", default=0.08, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--time_sec", default=10, type=float)
parser.add_argument("--pi_n_neurons", default=500, type=int)
parser.add_argument("--ssp_dim", default=55, type=int)
parser.add_argument("--grid_cells", default=True, type=bool)
parser.add_argument("--number_gcs", default=1000, type=int)
parser.add_argument("--length_scale", default=0.1, type=float)
parser.add_argument("--error_corrector", default=0.1, type=float)
parser.add_argument("--n_corrector", default=1000, type=int)
parser.add_argument("--plot", default=True, type=bool)
parser.add_argument("--model", default="basic", type=str)

args = parser.parse_args()

time_sec = args.time_sec
dt = 0.001
timesteps = np.arange(0, time_sec, dt)

domain_dim = args.domain_dim
ssp_dim = args.ssp_dim
length_scale = args.length_scale
radius = 1

path = np.hstack([
    nengo.processes.WhiteSignal(time_sec, high=args.limit, seed=1+i).run(time_sec,dt=dt) for i in range(domain_dim)
    # nengo.processes.WhiteSignal(time, high=args.limit, seed=args.seed+i) for i in range(domain_dim)
])
pathlen = path.shape[0]

# velocities!
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

pi_n_neurons = args.pi_n_neurons
gc_n_neurons = args.number_gcs
tau = 0.05
n_timesteps = timesteps.shape[0]

model = nengo.Network(seed=1)
with model:
    vel_input = nengo.Node(lambda t : vels_scaled[int(np.minimum(np.floor(t/dt), n_timesteps-1))])
    # vel_input = nengo.Node(lambda t : [0,0])
    init_state = nengo.Node(lambda t : real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))]  if t<0.05 else np.zeros(d) )


    if args.model == "basic":
        print('basic')
        pathintegrator = get_path_integrator(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True)
    elif args.model == "triple":
        print("triple")
        pathintegrator = get_path_integrator_triple(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                  n_correctors=args.n_corrector, error_correction_factor=args.error_corrector, stable=True)
    elif args.model == "phase":
        print("phase")
        pathintegrator = get_path_integrator_phase(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                  n_correctors=args.n_corrector, error_correction_factor=args.error_corrector, stable=True)
    elif args.model == "shift":
        print("shift")
        pathintegrator = get_path_integrator_shift(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, 
                error_correction_factor=args.error_corrector, stable=True)
    else:
        pathintegrator = get_path_integrator(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, with_gcs=True, n_gcs=gc_n_neurons, stable=True)
        
    nengo.Connection(vel_input, pathintegrator.velocity_input, synapse=None)
    nengo.Connection(init_state, pathintegrator.input, synapse=None)

    ssp_probe = nengo.Probe(pathintegrator.output, synapse=0.05)
    print(pathintegrator.output.neurons.probeable)
    print(pathintegrator.output.probeable)
    spike_probe = nengo.Probe(pathintegrator.output.neurons, synapse=0.05)
        
    
start = time.time()
with nengo.Simulator(model) as sim:
   sim.run(args.time_sec)

running_time = time.time()-start

print(sim.data[ssp_probe].shape)
print(sim.data[spike_probe].shape)
# samples = ssp_space.get_sample_pts_and_ssps(method='grid', 
#                         num_points_per_dim=100)
# sim_path_est = np.zeros((sim.data[ssp_probe].shape[0], 2))
# sim_path_est[0:60000,:]  = ssp_space.decode(sim.data[ssp_probe][0:60000], 'from-set','grid', 100, samples=samples)
# sim_path_est[60000:,:]  = ssp_space.decode(sim.data[ssp_probe][60000:], 'from-set','grid', 100, samples=samples)

# print("vs")

sim_path_est = ssp_space.decode(sim.data[ssp_probe], 'from-set','grid', 100)
print(sklearn.metrics.root_mean_squared_error(sim_path_est, path))
one_neuron = sim.data[spike_probe][:,30]
print(np.unique(one_neuron, return_counts=True))
spike = one_neuron > 0
print(spike.shape)
peaks = scipy.signal.find_peaks(one_neuron, distance=100)
print(peaks)
    
if args.plot:
    # Plot estimate
    fig = plt.figure(figsize=(5.5, 3.5),dpi=200)
    spec = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.plot(sim.trange(), np.sqrt(np.sum(sim.data[ssp_probe]*real_ssp,axis=1))/np.linalg.norm(sim.data[ssp_probe],axis=1))
    ax0.set_ylabel("Similarity")
    ax0.set_xlabel("Time (s)")
    ax0.set_xlim([0,time_sec])
    plt.legend()
    ax10 = fig.add_subplot(spec[1, 0])
    ax10.plot(sim.trange(), path[:,0],color='gray')
    ax10.plot(sim.trange(), sim_path_est[:,0],'--',color='k')
    ax10.set_xlim([0,time_sec])
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('x')
    ax11 = fig.add_subplot(spec[1, 1])
    ax11.plot(sim.trange(), path[:,1],color='gray')
    ax11.plot(sim.trange(), sim_path_est[:,1],'--',color='k')
    ax11.set_xlim([0,time_sec])
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('y')
    fig.suptitle('PI output')

    # plt.figure(dpi=200)
    # plt.title("spiking")
    # plt.plot(sim.trange(), one_neuron)
    # plt.show()
    
    plt.figure(dpi=200)
    plt.title('Rover trajectory in environment')
    plt.plot(path[:,0], path[:,1],color='gray', label='Ground truth')
    plt.plot(sim_path_est[10:-1000,0], sim_path_est[10:-1000,1],'--', label='Output of PI', lw=2)
    grid_cell_spikes = path[spike]
    plt.scatter(grid_cell_spikes[:,0], grid_cell_spikes[:,1], color='red')
    plt.legend()
    plt.show()

    #plt.figure(dpi=200)
    #plt.title("spiking")
    #plt.plot(sim.trange(), one_neuron)
    #plt.show()

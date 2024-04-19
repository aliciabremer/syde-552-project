import numpy as np
import nengo
import matplotlib.pyplot as plt
# from ssp import HexSSP
from sspslam.sspspace import HexagonalSSPSpace

import argparse
import random
import sys, os
import os.path
import pytry
import argparse
from models.pathintegration import get_path_integrator, get_path_integrator_triple, get_path_integrator_phase, get_path_integrator_shift


class IntegratorTrial(pytry.Trial):
    def params(self):
        self.param("ssp_dim", ssp_dim=151)
        self.param("domain_dim", domain_dim=2)
        self.param("time_sec", time_sec=120)
        self.param("path_limit", limit=0.08)
        self.param("pi_n_neurons", pi_n_neurons=400)
        self.param("gc_n_neurons", gc_n_neurons=1000)
        self.param("coupling_method", coupling_method=None)

    def evaluate(self, p):
        domain_dim = p.domain_dim
        bounds = np.tile([-1,1], (domain_dim,1))
        ssp_space = HexagonalSSPSpace(
            domain_dim=domain_dim,
            ssp_dim=p.ssp_dim,
            domain_bounds=1.1*bounds,
            length_scale=0.1
        )

        d = ssp_space.ssp_dim

        time_sec = p.time_sec
        dt = 0.001

        pi_n_neurons = p.pi_n_neurons
        gc_n_neurons = p.gc_n_neurons
        coupling_method = p.coupling_method

        tau = 0.05
        model = nengo.Network(seed=p.seed)
        with model:
            vel_input = nengo.Node(lambda t : np.zeros(domain_dim), label='vel_input')
            init_state = nengo.Node(lambda t : \
                np.zeros(d),
                label='init_state')
            
            if coupling_method == "get_path_integrator_triple":
                pathintegrator = get_path_integrator_triple(
                    ssp_space,
                    pi_n_neurons,
                    tau,
                    with_gcs=True,
                    n_gcs=gc_n_neurons,
                    n_correctors=1000,
                    error_correction_factor=0.1,
                    stable=True
                )
            elif coupling_method == "get_path_integrator_phase":
                pathintegrator = get_path_integrator_phase(
                    ssp_space,
                    pi_n_neurons,
                    tau,
                    with_gcs=True,
                    n_gcs=gc_n_neurons,
                    n_correctors=1000,
                    error_correction_factor=0.1,
                    stable=True
                )
            elif coupling_method == "get_path_integrator_shift":
                pathintegrator = get_path_integrator_shift(
                    ssp_space,
                    pi_n_neurons,
                    tau,
                    with_gcs=True,
                    n_gcs=gc_n_neurons,
                    n_correctors=1000,
                    error_correction_factor=0.1,
                    stable=True
                )
            else:
                pathintegrator = get_path_integrator(
                    ssp_space,
                    pi_n_neurons,
                    tau, 
                    with_gcs=True,
                    n_gcs=gc_n_neurons,
                    stable=True
                )
        
            nengo.Connection(vel_input, pathintegrator.velocity_input, synapse=None)
            nengo.Connection(init_state, pathintegrator.input, synapse=None)

            ssp_probe = nengo.Probe(pathintegrator.output, synapse=0.05)


        sim = nengo.Simulator(model)
        with sim:
            sim.run(time_sec)
            
        ssp_path  = ssp_space.decode(sim.data[ssp_probe], 'from-set','grid', 100)

        return dict(
            ts=sim.trange(),
            ssp_space=ssp_space,
            output_ssp_probe = sim.data[ssp_probe],
            sim_path = ssp_path,
        )

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ssp_dim', type=int, default=151)
    parser.add_argument('--domain_dim', type=int, default=2)
    parser.add_argument('--pi_n_neurons', type=int, default=700)
    parser.add_argument('--gc_n_neurons', type=int, default=1000)
    parser.add_argument('--trial_time', type=float, default=120)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='data')

    n_neurons_vco = [300, 500, 700, 900]
    coupling = ["get_path_integrator_triple", "get_path_integrator_phase", "get_path_integrator_shift"]
    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        for cp in coupling:
            params = {
                'pi_n_neurons': args.pi_n_neurons,
                'gc_n_neurons':args.gc_n_neurons,
                'data_format':'npz',
                'data_dir':data_path,
                'seed':seed, 
                'ssp_dim':args.ssp_dim,
                'domain_dim':args.domain_dim,
                'time_sec': args.trial_time,
                'coupling_method': cp
                }
            r = IntegratorTrial().run(**params)


        

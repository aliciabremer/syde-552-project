import nengo
import numpy as np
import scipy
from utils import sparsity_to_x_intercept, get_to_Fourier, get_from_Fourier

# Adapted from sspslam.networks.PathIntegration
# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/main/sspslam/networks/pathintegration.py
# Implemented for learning purposes

def get_path_integrator(ssp_space, n_neurons_pi, recurrent_tau=0.05,
                        scaling_factor=1, stable=True, max_radius=1,
                        with_gcs=True, n_gcs=1000, solver_weights=False,
                        label='pathint', intercepts=None, encoders=None,
                        **kwargs):
    if stable == True:
        def feedback(x):
            w = (x[2]/scaling_factor)/ssp_space.length_scale[0] # frequency of oscillation
            r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
            dx0 = x[0]*(max_radius**2-r**2)/r - x[1]*w
            dx1 = x[1]*(max_radius**2-r**2)/r + x[0]*w
            out = np.array([
                recurrent_tau*dx0 + x[0],
                recurrent_tau*dx1 + x[1],
                [0]
            ]).flatten()
            return out

    else:
        def feedback(x):
            w = (x[2]/scaling_factor)/ssp_space.length_scale[0] # frequency of oscillation
            dx0 = - x[1] * w # d Re F when no attractor
            dx1 = x[0] * w # d Im F when no attractor
            out = np.array([
                recurrent_tau * dx0 + x[0],
                recurrent_tau* dx1 + x[1],
                [0]
            ]).flatten()
            return out
        
    d = ssp_space.ssp_dim
    N = ssp_space.domain_dim
    n_oscs = (d+1)//2

    to_ssp = get_from_Fourier(d)
    to_Fourier = get_to_Fourier(d)
    
    net = nengo.Network(label=label,)
    with net:
        net.velocity_input = nengo.Node(label=label+'_velocity_input', size_in = N)
        net.input = nengo.Node(label=label+'_input', size_in=d)
        if intercepts is not None and encoders is not None:
            net.output = nengo.Ensemble(
                n_gcs,d,encoders=encoders,
                intercepts=intercepts,
                label=label+'_output')
        elif with_gcs:
            encoders = ssp_space.sample_grid_encoders(n_gcs)
            net.output = nengo.Ensemble(
                n_gcs,d,encoders=encoders,
                intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]),
                label=label+'_output')
        else:
            net.output = nengo.Node(label=label+'_output', size_in = d)

        # each VCO is an oscillator
        net.oscillators = nengo.networks.EnsembleArray(
            n_neurons_pi, n_oscs, ens_dimensions=3,
            radius=np.sqrt(2), label=label+"_vco", **kwargs
        )
        net.oscillators.output.output = lambda t, x : x

        # input
        nengo.Connection(net.input,net.oscillators.input, transform=to_Fourier)
        
        # oscillators
        for i in range(1, n_oscs):
            nengo.Connection(net.velocity_input, net.oscillators.ea_ensembles[i],
                            transform = np.vstack([np.zeros((2,N)), ssp_space.phase_matrix[i,:].reshape(1,-1)]),
                            synapse = recurrent_tau)
            
            nengo.Connection(net.oscillators.ea_ensembles[i], net.oscillators.ea_ensembles[i],
                            function=feedback, synapse=recurrent_tau, solver=nengo.solvers.LstsqL2(weights=solver_weights))
            
        # dc term constant
        zerofreq = nengo.Node([1,0,0],label=label+'_zerofreq')
        nengo.Connection(zerofreq, net.oscillators.ea_ensembles[0], synapse=None)
        nengo.Connection(net.oscillators.output, net.output, transform=to_ssp)

    return net

# Adapted from sspslam.networks.PathIntegration_BCs_GCs
# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/main/sspslam/networks/pathintegration.py
# Slightly modified

def get_path_integrator_triple(ssp_space, n_neurons_pi, recurrent_tau=0.05,
                        scaling_factor=1, stable=True, max_radius=1,
                        with_gcs=True, n_gcs=1000, solver_weights=False,
                        n_correctors=1000, error_correction_factor = 0.1,
                        label='pathint', intercepts=None, encoders=None,
                         **kwargs):
    net = get_path_integrator(ssp_space, n_neurons_pi,
                              recurrent_tau=recurrent_tau,
                              scaling_factor=scaling_factor,
                              stable=stable,
                              max_radius=max_radius,
                              with_gcs=True,
                              n_gcs=n_gcs,
                              intercepts=intercepts,
                              encoders=encoders, **kwargs)
        
    def correction_feedback0(x):
        er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
        er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
        res1 = (er_r0 + 1j*er_i0)**(1/3)
        er_r = res1.real
        er_i = res1.imag
        div = er_r**2+er_i**2 # math suggests dividing by this
        if np.isclose(div, 0, atol=1e-4): # ensure not close to 0... might also make sense to check < something not sure
            return  x[:2]
        else:
            res = np.array([er_r*x[0]+er_i*x[1], er_r*x[1] - er_i*x[0]]) / div
            return (error_correction_factor)*(res - x[:2]) + x[:2]
         
    def correction_feedback1(x):
        er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
        er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
        res1 = (er_r0 + 1j*er_i0)**(1/3)
        er_r = res1.real
        er_i = res1.imag
        div = er_r**2+er_i**2
        if np.isclose(div, 0, atol=1e-4):
            return  x[2:4]
        else:
            res = np.array([er_r*x[2]+er_i*x[3], er_r*x[3] - er_i*x[2]]) / div
            return (error_correction_factor)*(res - x[2:4]) + x[2:4]
         
    def correction_feedback2(x):
        er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
        er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
        res1 = (er_r0 + 1j*er_i0)**(1/3)
        er_r = res1.real
        er_i = res1.imag
        div = er_r**2+er_i**2
        if np.isclose(div, 0, atol=1e-4):
            return  x[4:6]
        else:
            res = np.array([er_r*x[4]+er_i*x[5], er_r*x[5] - er_i*x[4]]) / div
            return (error_correction_factor)*(res - x[4:]) + x[4:]
        
    d = ssp_space.ssp_dim
    N = ssp_space.domain_dim
    n_oscs = (d+1)//2
        
    net.corrector = nengo.networks.EnsembleArray(n_correctors, n_oscs//3 , 
                                                              ens_dimensions = 6,
                                                              radius=np.sqrt(2), label=label+"_corrector")
        
    net.corrector.output.output = lambda t, x : x

    # connect corrector to oscillators
    for i in range(1, n_oscs):
        nengo.Connection(net.oscillators.ea_ensembles[i][:2], 
                                net.corrector.ea_ensembles[int((i-1)//3)][2*np.mod(i-1,3) + np.array([0,1])],
                                synapse=recurrent_tau)
        
    for i in range(int(n_oscs//3)):
            nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+1][:2], 
                             function=correction_feedback0,synapse=recurrent_tau)
            nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+2][:2], 
                             function=correction_feedback1,synapse=recurrent_tau)
            nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+3][:2], 
                             function=correction_feedback2,synapse=recurrent_tau)
    
    return net


# implementing the tangent coupling method
def get_path_integrator_phase(ssp_space, n_neurons_pi, recurrent_tau=0.05,
                        scaling_factor=1, stable=True, max_radius=1,
                        with_gcs=True, n_gcs=1000, solver_weights=False,
                        n_correctors=1000, error_correction_factor = 0.1,
                        intercepts=None, encoders=None,
                        label='pathint', **kwargs):
    
    net = get_path_integrator(ssp_space, n_neurons_pi,
                              recurrent_tau=recurrent_tau,
                              scaling_factor=scaling_factor,
                              stable=stable,
                              max_radius=max_radius,
                              with_gcs=True,
                              n_gcs=n_gcs,
                              intercepts=intercepts,
                              encoders=encoders, **kwargs)
    
    # correct (should probably be able to compute phase difference better)
    def correction_feedback(x):
        est = -np.arctan2(x[0] * x[3] + x[1]*x[2], x[0]*x[3]-x[1]*x[3])
        act = np.arctan2(x[5],x[4])
        diff = np.exp(1.j*error_correction_factor*(est-act))
        new = np.array([x[4] * diff.real - x[5] * diff.imag, x[4] * diff.imag + x[5] * diff.real])
        return new
    
    d = ssp_space.ssp_dim
    N = ssp_space.domain_dim
    n_oscs = (d+1)//2
    net.corrector = nengo.networks.EnsembleArray(n_correctors,
                                                n_oscs//3, 
                                                ens_dimensions = 6,
                                                radius=np.sqrt(2),
                                                label=label+"_corrector")
    
    # connect to oscillators
    for i in range(1, n_oscs):
        nengo.Connection(net.oscillators.ea_ensembles[i][:2], 
                        net.corrector.ea_ensembles[int((i-1)//3)][2*np.mod(i-1,3) + np.array([0,1])],
                        synapse=recurrent_tau)
            
    for i in range(int(n_oscs//3)):
        nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+3][:2],
                        function=correction_feedback,synapse=recurrent_tau)
    return net


# phase shifting
def get_path_integrator_shift(ssp_space, n_neurons_pi, recurrent_tau=0.05,
                        scaling_factor=1, stable=True, max_radius=1,
                        with_gcs=True, n_gcs=1000, solver_weights=False,
                        n_correctors=1000, error_correction_factor = 0.1,
                        intercepts=None, encoders=None,
                        label='pathint', **kwargs):

    net = get_path_integrator(ssp_space, n_neurons_pi,
                              recurrent_tau=recurrent_tau,
                              scaling_factor=scaling_factor,
                              stable=stable,
                              max_radius=max_radius,
                              with_gcs=True,
                              n_gcs=n_gcs,
                              intercepts=intercepts,
                              encoders=encoders, **kwargs)
        
    def compute_cs(x):
        c1 = x[0]*x[2] + x[1]*x[3]
        s1 = x[0]*x[3] - x[1]*x[2]

        c2 = x[2]*x[4] + x[3]*x[5]
        s2 = x[2]*x[5] - x[3]*x[4]

        c3 = x[4]*x[0] + x[5]*x[1]
        s3 = x[4]*x[1] - x[5]*x[0]

        return [(c1+c2+c3)/3, (s1+s2+s3)/3]


    def correction_feedback0(x):
        aest = x[2]*x[6] + x[3]*x[7]
        best = -x[2]*x[7] + x[3]*x[6]
        adiff = aest - x[0]
        bdiff = best - x[1]
        # also compute backwards diff
        alphest = x[4]*x[6] - x[5]*x[7]
        betaest = x[4]*x[7] + x[5]*x[6]
        alphadiff = alphest - x[0]
        betadiff = betaest - x[1]

         # use error correction factor for both (could also divide by both)
        return [
            x[0] + adiff * error_correction_factor + alphadiff*error_correction_factor,
            x[1] + bdiff*error_correction_factor + betadiff *error_correction_factor
        ]

         
    def correction_feedback1(x):
        aest = x[4]*x[6] + x[5]*x[7]
        best = -x[4]*x[7] + x[5]*x[6]
        adiff = aest - x[2]
        bdiff = best - x[3]
        # also compute backwards diff
        alphest = x[0]*x[6] - x[1]*x[7]
        betaest = x[0]*x[7] + x[1]*x[6]
        alphadiff = alphest - x[2]
        betadiff = betaest - x[3]

        return [
            x[2] + adiff * error_correction_factor + alphadiff*error_correction_factor,
            x[3] + bdiff*error_correction_factor + betadiff *error_correction_factor
        ]

         
    def correction_feedback2(x):
        aest = x[0]*x[6] + x[1]*x[7]
        best = -x[0]*x[7] + x[1]*x[6]
        adiff = aest - x[4]
        bdiff = best - x[5]
        # also compute backwards diff
        alphest = x[2]*x[6] - x[3]*x[7]
        betaest = x[2]*x[7] + x[3]*x[6]
        alphadiff = alphest - x[4]
        betadiff = betaest - x[5]

        return [
            x[4] + adiff * error_correction_factor + alphadiff*error_correction_factor,
            x[5] + bdiff*error_correction_factor + betadiff *error_correction_factor
        ]
        
    d = ssp_space.ssp_dim
    N = ssp_space.domain_dim
    n_oscs = (d+1)//2

    net.correctors = nengo.networks.EnsembleArray(n_correctors, n_oscs//3 , 
                                                              ens_dimensions = 8,
                                                              radius=np.sqrt(2), label=label+"_corrector")
        
    net.correctors.output.output = lambda t, x : x


    # connect
    for i in range(1, n_oscs):
        nengo.Connection(net.oscillators.ea_ensembles[i][:2], 
                        net.corrector.ea_ensembles[int((i-1)//3)][2*np.mod(i-1,3) + np.array([0,1])],
                        synapse=recurrent_tau)
        
    for i in range(int(n_oscs//3)):
        nengo.Connection(net.corrector.ea_ensembles[i], 
                        net.corrector.ea_ensembles[i][6:],
                        function=compute_cs, synapse=recurrent_tau)
                
        nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+1][:2], 
                                 function=correction_feedback0,synapse=recurrent_tau)
        nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+2][:2], 
                                 function=correction_feedback1,synapse=recurrent_tau)
        nengo.Connection(net.corrector.ea_ensembles[i], net.oscillators.ea_ensembles[3*i+3][:2], 
                                 function=correction_feedback2,synapse=recurrent_tau)

    return net



# attempt to phase shift more
# doesn't work because VCOs not equally spaced out...
def OLD_get_path_integrator_total(ssp_space, n_neurons_pi, recurrent_tau=0.05,
                            scaling_factor=1, stable=True, max_radius=1,
                            with_gcs=True, n_gcs=1000, solver_weights=False,
                            n_phase_steps=500, error_correction=0.2, num_phase_connected=10,
                            intercepts=None, encoders=None,
                            label='pathint', **kwargs):
    
    net = get_path_integrator(ssp_space, n_neurons_pi,
                              recurrent_tau=recurrent_tau,
                              scaling_factor=scaling_factor,
                              stable=stable,
                              max_radius=max_radius,
                              with_gcs=True,
                              n_gcs=n_gcs,
                              intercepts=intercepts,
                              encoders=encoders, **kwargs)
        
    def compute_cs(x):
        c = x[0]*x[2] + x[1]*x[3]
        s = -x[0]*x[3] + x[1]*x[2]
        return [ c, s]

    def phase_step_connection(x):
        # calculate
        return np.mean(x[::2]), np.mean(x[1::2])
    
    def correction_feedback0(x):
        # calculate
        aest = x[2]*x[6] + x[3]*x[7]
        best = -x[2]*x[7] + x[3]*x[6]
        adiff = aest - x[0]
        bdiff = best - x[1]
        return [x[0] + adiff * error_correction, x[1] + bdiff*error_correction]

    def correction_feedback1(x):
        alphest = x[0]*x[6] + x[1]*x[7]
        betaest = x[0]*x[7] + x[1]*x[6]
        alphadiff = alphest - x[0]
        betadiff = betaest - x[1]
        return [x[2] + alphadiff * error_correction, x[3] + betadiff*error_correction]


    d = ssp_space.ssp_dim
    N = ssp_space.domain_dim
    n_oscs = (d+1)//2
    n_phase_step = n_oscs-2

    net.phase_step = nengo.networks.EnsembleArray(n_phase_steps, n_oscs-2, 
                                                ens_dimensions = 8+2*num_phase_connected,
                                                radius=np.sqrt(2), label=label+"_corrector")
        
    net.phase_step.output.output = lambda t, x : x

    for i in range(0, n_oscs-2):
        nengo.Connection(net.oscillators.ea_ensembles[i+1][:2],
                                   net.phase_step.ea_ensembles[i][0:2],
                                   synapse = recurrent_tau)
                
        nengo.Connection(net.oscillators.ea_ensembles[i+1][:2],
                                   net.phase_step.ea_ensembles[i][2:4],
                                   synapse = recurrent_tau)

        nengo.Connection(net.phase_step.ea_ensembles[i][0:4],
                                   net.phase_step.ea_ensembles[i][4:6],
                                   function=compute_cs, synapse = recurrent_tau)
                
        subset = np.random.permutation(n_oscs-2)[0:num_phase_connected]
        subset[0] = i

        for j in range(num_phase_connected):
            nengo.Connection(net.phase_step.ea_ensembles[int(subset[j])][4:6], # FIX WEIGHTS
                                    net.phase_step.ea_ensembles[i][8+j*2:8+(j+1)*2],
                                    synapse = recurrent_tau)
                    
        nengo.Connection(net.phase_step.ea_ensembles[i][8:],
                            net.phase_step.ea_ensembles[i][6:8],
                           function=phase_step_connection, synapse = recurrent_tau)
        
    # feedback
    for i in range(n_oscs-2):
        nengo.Connection(net.phase_step.ea_ensembles[i], net.oscillators.ea_ensembles[i+1][:2], 
                                 function=correction_feedback0,synapse=recurrent_tau)
        nengo.Connection(net.phase_step.ea_ensembles[i], net.oscillators.ea_ensembles[i+2][:2], 
                                 function=correction_feedback1,synapse=recurrent_tau)

    return net

# utils from sspslam that are not exported
import nengo
import numpy as np
import scipy


# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/new/sspslam/networks/pathintegration.py

# matrix for oscillator format -> ssp
def get_from_Fourier(d):
    k = (d+1) // 2
    if d%2 == 0:
        shiftmat = np.zeros((4*k, 3*k))
        shiftmat[:k,0::3] = np.eye(k)
        shiftmat[k,0] = 1
        shiftmat[k+1:2*k,3::3] = np.flip(np.eye(k-1),axis=0)
        shiftmat[2*k:3*k,1::3] = np.eye(k)
        shiftmat[2*k,1] = 0
        shiftmat[3*k+1:,4::3] = -np.flip(np.eye(k-1), axis=0)
    else:
        shiftmat = np.zeros((4*k-2,3*k))
        shiftmat[:k,0::3] = np.eye(k)
        shiftmat[k:2*k-1,3::3] = np.flip(np.eye(k-1), axis=0)
        shiftmat[2*k-1:3*k-1,1::3] = np.eye(k)
        shiftmat[3*k-1:,4::3] = -np.flip(np.eye(k-1), axis=0)
    
    invW = np.fft.ifft(np.eye(d))
    M = np.hstack([invW.real, -invW.imag]) @ shiftmat
    return M

# matrix for ssp -> oscillator format
def get_to_Fourier(d):
    k = (d+1)//2
    M = np.zeros((3*k,d))
    M[3:-1:3,:] = np.fft.fft(np.eye(d))[1:k,:].real
    M[4::3,:] = np.fft.fft(np.eye(d))[1:k:].imag
    return M

# https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/new/sspslam/utils/utils.py


def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


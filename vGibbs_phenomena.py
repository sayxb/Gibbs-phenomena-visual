import numpy as np
import matplotlib.pyplot as plt

dx = 0.001 # Resolution
L = 2*np.pi
x = np.arange(0, L+dx, dx)
xquart = int(np.floor(len(x)/4))
k_range = 225

# Defining Hat Function

f = np.zeros_like(x)
f[xquart:3*xquart] = 1

A0 = 2/L * np.sum(f * np.ones_like(x)) * dx
k_vals = np.arange(1, k_range + 1)

eix = np.cos(np.outer(k_vals, x)) + np.sin(np.outer(k_vals,x))*(1j)
A_k = 2/L * np.dot(np.real(eix), f) * dx
B_k = 2/L * np.dot(np.imag(eix), f) * dx

F_S = A0/2  + np.cumsum(A_k[:, np.newaxis] * np.real(eix), axis = 0) + np.cumsum(B_k[:, np.newaxis] * np.imag(eix), axis=0)

maxlist = np.ones(k_range)
minlist = np.ones(k_range)
abs_sumlist = np.ones(k_range)
lisbasis = np.linspace(0, k_range, k_range)


for i in range(0, k_range):
    maxlist[i] = max(F_S[i])
    minlist[i] = min(F_S[i])
    abs_sumlist[i] = maxlist[i] - abs(minlist[i]) - 1
    
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(x, f, color='k')
    ax[0,0].plot(x, F_S[i], color='r', lw=0.8)
    ax[0,0].set_title(f'{i + 1} terms estimation')
    
    ax[0,1].plot(lisbasis, maxlist)
    ax[0,1].set_title('Max Amplitudes')
    ax[0,1].set_xlabel('K (no. of terms)')
    ax[0,1].set_ylim(1.08, 1.14)
    
    ax[1,0].plot(lisbasis, minlist)
    ax[1,0].set_title('Min Amplitudes')
    ax[1,0].set_xlabel('K (no. of terms)')    
    ax[1,0].set_ylim(-0.1, -0.08)
    
    ax[1,1].plot(abs_sumlist, color='r')
    ax[1,1].set_ylim(-0.00015, 0.00015)
    ax[1,1].set_title('Max - (|Min| + 1)')
    ax[1,1].set_xlabel('K (no. of terms)')
    
    
    plt.tight_layout()
    plt.pause(0.001)
    
plt.show()


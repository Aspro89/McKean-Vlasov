#%% Import packages

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from McKean_Vlasov import energy, entropy2, IC, interaction_energy, path, Wasserstein
from scipy.fft import rfft
from scipy.stats import norm, vonmises
import time
plt.rc('font', size=18) #controls default text size

# 'r': recursion; 'e': energy minimisation; 't': temporal empirical measure
mode = 'c'
# 'e': euclidean; 't': torus
modeBC = 't'
# Initial conditions
# 'u': uniform; 'n': unimodal; 'b': bimodal; 'b2': closer bimodal; 'up': perturbed uniform
# distrs = ['n', 'b', 'u']
distrs = ['n']


#%% Parameters

sigma = .1

from McKean_Vlasov import Morse_force, Morse_potential
l = 1/10
C = 0
r = .5
def F(x):
    return Morse_potential(x, L, l, C, r, modeBC)
def nabla_F(x):
    return Morse_force(x, L, l, C, r, modeBC)

# from McKean_Vlasov import Kuramoto_force, Kuramoto_potential
# K = .1
# n = 10
# def F(x):
#     return Kuramoto_potential(x, K, L, n)
# def nabla_F(x):
#     return Kuramoto_force(x, K, L, n)


L = 2*np.pi  # range of state space
bins = np.linspace(-L/2, L/2, 51) # bins for histograms

dx = .0001        # space step
X = np.arange(-L/2, L/2, dx)
# Nx = 100        # number of space intervals
# dx = L / Nx     # space step for finite difference method
# X = np.linspace(-L/2, L/2, Nx)  # state space vector

if mode == 'r':
    num = len(distrs)
elif mode == 'e':
    num = 1
else:
    num = 0
fig, ax = plt.subplots(num+2, 1)
fig.set_size_inches(15, 20, forward=True)
# fig.suptitle(r'$a = $' + str(a) + r', $b = $' + str(b) + r', $c = $' + str(c) + r', $\sigma = $' + str(sigma), fontsize='x-large')

ax[0].plot(X, F(X), color='b', label=r'$F$')
ax[0].legend()
ax[1].plot(X[:len(X)//2], -nabla_F(X[:len(X)//2]), color='b', label=r'$-\nabla F$')
ax[1].plot(X[len(X)//2+1:], -nabla_F(X[len(X)//2+1:]), color='b')
ax[1].legend()


#%% Cluster number

if mode == 'c':
    fig2, ax2 = plt.subplots(2, 1)
    fig2.set_size_inches(15, 20, forward=True)
    gamma = 1j / L * rfft(nabla_F(X), 1000)
    k = 2*np.pi/L * np.arange(len(gamma))
    gamma = np.real(k * gamma)# - sigma**2 * k**2 / 2
    ax2[0].plot(X[:len(X)//2], -nabla_F(X[:len(X)//2]), color='b', label=r'$-\nabla F$')
    ax2[0].plot(X[len(X)//2+1:], -nabla_F(X[len(X)//2+1:]), color='b')
    ax2[1].plot(k, gamma)
    kmax = np.argmax(gamma)
    period = 2*np.pi/kmax
    ncluster = L//period
    print(period, ncluster)
    

#%% Fixed-point calculation

if mode == 'r':
    # NV = [V(distance(X[i] - X, L, modeBC), a, b, c) for i, _ in enumerate(X)]
    if modeBC == 't':
        NV = np.flip(np.roll(F(X), -len(X)//2-1))
    else:
        NV = np.flip(np.roll(F(np.arange(-L, L, dx)), -len(X)-1))
    
    maxiter = 30
    
    for count, distr in enumerate(distrs):
        
        rho0 = IC(L, dx, 1, distr, 't', 'd')
        # logrho0 = np.log(rho0)
        ax[2+count].plot(X, rho0, label=r'$c_0$')
        rho = deepcopy(rho0)
        # logrho = deepcopy(logrho0)
        
        # rho1 = np.abs(np.empty_like(rho))
        rho1 = np.zeros_like(rho)
        rho1[0] = 1
        niter = 0
        count2 = 0
        M = np.max(rho0)
        print('\nDistribution = ' + distr)
        print('# Wasserstein_distance energy')
        while Wasserstein(X, X, rho1, rho, L, modeBC) > 1e-3 and niter < maxiter:
            rho1 = deepcopy(rho)
            if modeBC == 't':
                # u = [np.exp(-2/sigma**2*np.sum(NV[j] * rho) * dx) for j, _ in enumerate(X)]
                u = [np.exp(-2/sigma**2*np.sum(np.roll(NV, j) * rho) * dx) for j, _ in enumerate(X)]
                rho = u/np.sum(u)/dx
            else:
                # u = [np.exp(-2/sigma**2*np.trapz(NV[j] * rho, dx=dx)) for j, _ in enumerate(X)]
                u = [np.exp(-2/sigma**2*np.trapz(np.roll(NV, j)[:len(X)] * rho, dx=dx)) for j, _ in enumerate(X)]
                rho = u/np.trapz(u, dx=dx)
            
            print('{:<3}'.format(niter+1) + '{:^19.4f}'.format(Wasserstein(X, X, rho1, rho, L, modeBC)), '{:.4f}'.format(energy(rho, L, dx, F, sigma, modeBC)))
            niter += 1
            
            # for i in range(niter):
            #     u = [-2/sigma**2*np.sum(np.roll(NV, j) * np.exp(logrho)) * dx for j, _ in enumerate(X)]
            #     logrho = u - np.log(np.sum(np.exp(u)) * dx)
            # rho = np.exp(logrho)
            if np.max(rho) > M:
                M = np.max(rho)
            if Wasserstein(X, X, rho1, rho, L, modeBC) < 1e-3 or niter >= maxiter-1:
                ax[2+count].plot(X, rho, label=r'$\bar c_{{{}}}$'.format(count2+1))
                count2 += 1
            ax[2+count].axis([-L/2, L/2, 0, 1.1*M])
            ax[2+count].legend()
            
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    # filename = 'Figures/Invariant_measure_' + f'{a=}'.split('=')[0] + str(a) + '_' + f'{b=}'.split('=')[0] + str(b) + '_' + f'{c=}'.split('=')[0] + str(c) + '_' + 'σ' + str(sigma) + ".png"
    # fig.savefig(filename, format="png")


#%%% Energy minimisation

if mode == 'e':
    
    def distr(v):
        if modeBC == 't':
            return vonmises(1/v, scale=L/2/np.pi)
        else:
            return norm(0, v)
    
    def Ent(v):
        return entropy2(distr(v).pdf(X), int(L/dx), modeBC)
    def En2(v):
        return interaction_energy(distr(v).pdf(X), L, dx, F, modeBC)
    def En(v):
        return energy(distr(v).pdf(X), L, dx, F, sigma, modeBC)
    
    ss = np.arange(.01, 1.5, .01)
    E = [En(i) for i in ss]
    
    if modeBC == 't':
        xlabel=r'$\kappa^{-1}$'
    else:
        xlabel=r'$\sigma^2$'
    
    fig2, ax2 = plt.subplots(3, 1)
    fig2.set_size_inches(15, 20, forward=True)
    ax2[0].plot(ss, [Ent(i) for i in ss])
    ax2[0].set(xlabel=xlabel, title='relative entropy')
    # from scipy.special import i0, i1
    # tt = 1/ss
    # ax2[0].plot(ss, tt * i1(tt) / i0(tt) - np.log(i0(tt)))
    ax2[1].plot(ss, [En2(i) for i in ss])
    ax2[1].set(xlabel=xlabel, title='interaction energy')
    ax2[2].plot(ss, E)
    ax2[2].set(xlabel=xlabel, title='total free energy')
    
    plt.figure(fig)
    ax[2].plot(X, distr(ss[np.argmin(E)]).pdf(X), label=r'$\bar c$')
    # rho0 = vonmises(1/ss[np.argmin(E)], scale=L/4/np.pi).pdf(X)/2
    # ax[2].plot(X, rho0, label=r'$\bar c$')
    # print(energy(rho0, L, dx, F, sigma, modeBC))
    ax[2].legend()
    plt.show()


#%%% Time sampling

if mode == 't':
    T = np.int16(5e1)  # final simulation time
    dt = np.float16((dx ** 2) / (2 * sigma ** 2))
    NT = int(T / dt)
    h = sigma*np.sqrt(dt)  # scaled time step for Euler-Maruyama = dx/np.sqrt(2)
    
    Nsample = np.int16(.9*T)
    N = 10
    
    for count, distr in enumerate(distrs):
        # Initial conditions
        Y0  = IC(L, dx, N, distr, modeBC, 'r')
        
        seed = 57  #57 #10 #3
        np.random.seed(seed)
        
        start_time = time.time()
        Y = path(NT, Nsample, dt, L, h, nabla_F, Y0, modeBC, 10)
        print("Total time: --- {:4.2f} s ---".format(time.time() - start_time))
        
        ax[2+count].hist(Y.flatten()[int(N*np.ceil(5/dt)):], bins=bins, density=True, label=r'$\mathcal{E}_t(X)$')
        # ax2.plot(Y)
        # ax.hist(Y.flatten()[N*NT//2:], bins=bins, density=True, label=r'$\mathcal{E}_t(X)$')
        fig.tight_layout()
        plt.show()
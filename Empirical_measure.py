#%% Import packages

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from McKean_Vlasov import clustering_distance, energy, energy_s, Euler_Maruyama, finite_differences, IC, position, Wasserstein
from scipy.stats import circmean, circstd
import time

plt.rc('font', size=18) # controls default text size

# 1 = single realisation; 'st' = stream; 'm' = many realisations; 's' = sigma sweep; 'r' = repulsion-intensity sweep
mode = 1
# 't' = torus, 'e' = Euclidean
modeBC = 't'
# Initial distribution
# 'u': uniform; 'n': unimodal; 'b': bimodal; 'b2': closer bimodal; 'up': perturbed uniform
distr = 'u'

save_fig = False
save_var = False


#%% Fixed parameters

from McKean_Vlasov import Morse_force, Morse_potential
def F(x):
    return Morse_potential(x, L, l, C, r, modeBC)
def nabla_F(x):
    return Morse_force(x, L, l, C, r, modeBC)

# from McKean_Vlasov import Kuramoto_force, Kuramoto_potential
# K = .1
# n = 2
# def F(x):
#     return Kuramoto_potential(x, K, L, n)
# def nabla_F(x):
#     return Kuramoto_force(x, K, L, n)

# T = 2265          # final simulation time
T = 145          # final simulation time
L = 2*np.pi     # range of state space (should be smaller than 2π)
Nbins = 50
bins = np.linspace(-L/2, L/2, Nbins+1) # bins for histograms

dx = .1        # space step
X = np.arange(-L/2, L/2, dx)
# Nx = 100        # number of space intervals
# dx = L / Nx     # space step for finite difference method
# X = np.linspace(-L/2, L/2, Nx)  # state space vector

N = 1000        # 1000 # number of particles

# possibly fix a seed
# seed = 5  # 57 # 10 # 3
# np.random.seed(seed)


#%%% Simulations    


#%% Single realisation

if mode == 1:
    l = 1/20
    C = 2
    r = 1
    sigma = .1
    dt = np.float16((dx ** 2) / (2 * sigma ** 2))   # time step for Euler-Maruyama and finite
                                                    # difference method, chosen according the
                                                    # Von Neumann stability condition: k * dt / dx**2 <= 1/2
    NT = int(T / dt)  # number of time steps
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7.5, forward=True)
    # title = '$T = $' + str(T) + ', $a = $' + str(a) + ', $b = $' + str(b) + ', $c = $' + str(c)
    # fig.suptitle(title, fontsize='x-large')
    
    u0 = IC(L, dx, N, distr, modeBC, 'd')
    Y0 = IC(L, dx, N, distr, modeBC, 'r')
    
    start_time = time.time()
    u = finite_differences(NT, dt, X, nabla_F, deepcopy(u0), modeBC)
    Y = Euler_Maruyama(NT, dt, L, nabla_F, deepcopy(Y0), modeBC, sigma)
    if save_var:
        np.save('Y.npy', Y)
    
    # plot of Euler-Maruyama + finite-difference simulation at t=0 and t=T
    ax.hist(Y0, bins=bins, density=True, label=r'$\rho^N(X(0))$', color='b', alpha=.3)
    ax.hist(Y, bins=bins, density=True, label=r'$\rho^N(X(T))$', color='r', alpha=.3)
    # print('Wasserstein distance = ' + str(Wasserstein(Y, X, np.ones_like(Y0), u, L, modeBC)))
    print('Energy of c_0        = ' + str(energy(u0, L, dx, F, sigma, modeBC)))
    print('Energy of c          = ' + str(energy(u, L, dx, F, sigma, modeBC)))
    print('Energy of rho^n_0    = ' + str(energy_s(Y0, L, F, sigma)))
    print('Energy of rho^n      = ' + str(energy_s(Y, L, F, sigma)))
    print("Total time: --- {:4.2f} s ---".format(time.time() - start_time))
    
    ax.plot(X, u0, label='$c(x,0)$', color='b')
    ax.plot(X, u, label='$c(x,T)$', color='r')
    ax.legend()
    
    fig.tight_layout()
    plt.show()


#%% Stream

if mode == 'st':
    l = 1/20
    C = 0
    r = 1
    sigma = .2
    
    from sklearn.cluster import DBSCAN
    def distance(x, y):
        return np.abs(position(x - y, L, 't'))
    model = DBSCAN(eps=.1, min_samples=30, metric=distance)
    
    Nplots = 5
    fig, ax = plt.subplots(Nplots)
    fig.set_size_inches(10, 20, forward=True)
    
    dt = np.float16((dx ** 2) / (2 * sigma ** 2))   # time step for Euler-Maruyama and finite
                                                    # difference method, chosen according the
                                                    # Von Neumann stability condition: k * dt / dx**2 <= 1/2
    NT = int(T / dt)  # number of time steps
    NTcluster = int(20//dt)
    tcluster = np.linspace(NTcluster, NT, 5, dtype=int)
    Y0 = IC(L, dx, N, distr, modeBC, 'r')
    Y = deepcopy(Y0)
    # Q = [clustering_distance(Y, distance)]
    start_time = time.time()
    for t in range(NTcluster):
        Y = Euler_Maruyama(1, dt, L, nabla_F, Y, modeBC, sigma)
        # Q = Q + [clustering_distance(Y, distance)]
    clustering = model.fit(Y.reshape(-1, 1))
    yhat = clustering.labels_
    clusters = np.unique(yhat)
    clusters = clusters[clusters!=-1]
    NC = len(clusters)
    # coarse-grained variable
    Z       = {'number': clusters, 'weight': np.empty(NC), 'mean': np.empty(NC), 'std': np.empty(NC)}
    for c, d in enumerate(clusters):
        cluster     = Y[yhat == d]
        # print(len(cluster)/len(Y))
        # weight, mean and standard deviation of each cluster
        Z['weight'][c]   = len(cluster)/len(Y)
        Z['mean'][c]     = circmean(cluster, -L/2, L/2)
        Z['std'][c]      = circstd(cluster, -L/2, L/2)
    i = 0
    for t in range(NTcluster, NT+1):
        Y = Euler_Maruyama(1, dt, L, nabla_F, Y, modeBC, sigma)
        if t in tcluster:
            ax[i].hist(Y0, bins=bins, density=True, label=r'$\rho^N(X(0))$', color='b', alpha=.3)
            ax[i].hist(Y, bins=bins, density=True, label=r'$\rho^N(X(T))$', color='r', alpha=.3)
            # ax[i].vlines(centers, 0, weights/L*Nbins)
            i += 1
    print("Total time: --- {:4.2f} s ---".format(time.time() - start_time))
    

#%% Many realisations
    
if mode == 'm':
    import multiprocessing
    l = 1/50
    C = .5
    r = .5
    sigma = .1
    dt = np.float16((dx ** 2) / (2 * sigma ** 2))   # time step for Euler-Maruyama and finite
                                                    # difference method, chosen according the
                                                    # Von Neumann stability condition: k * dt / dx**2 <= 1/2
    NT = int(T / dt)  # number of time steps
    
    u0 = IC(L, dx, N, distr, modeBC, 'd')
    u  = finite_differences(NT, dt, X, nabla_F, deepcopy(u0), modeBC)
    
    nsample = 100
    
    if __name__ == '__main__':
        
        W = []
        def collectW(result):
            global W
            W = W + [Wasserstein(result, X, np.ones_like(Y0), u, L, modeBC)]
            return W
        YY = []
        def collect(result):
            global W, YY, uu
            W = W + [Wasserstein(result, X, np.ones_like(Y0), u, L, modeBC)]
            YY = YY + [result]
            return W, YY
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        start_time = time.time()
        for k in range(nsample):
            Y0 = IC(L, dx, N, distr, modeBC, 'r')
            pool.apply_async(Euler_Maruyama, args = (NT, dt, L, nabla_F, deepcopy(Y0), modeBC, sigma, 10, k), callback=collect)
        pool.close()
        pool.join()
        print("Total time: --- {:4.2f} s ---".format(time.time() - start_time))
        
        fig, ax = plt.subplots(4)
        fig.set_size_inches(10, 25, forward=True)
        ax[-1].hist(W, density=True)
        ax[-1].set(title='Histogram for ' + r'$W_1(\rho^N(T), \rho(T))$')
        
        # Three cases
        ind = sorted(range(nsample), key=lambda k: W[k])
        plots = [ind[0], ind[int(len(ind)//2)], ind[-1]]
        # fig.suptitle(r'$T = $' + str(T) + r', $a = $' + str(a) + r', $b = $' + str(b) + r', $c = $' + str(c) + r', $\sigma = $' + str(sigma), fontsize='x-large')
        for j, k in enumerate(plots):
            ax[j].hist(Y0, bins=bins, density=True, label=r'$\rho^N(X(0))$', color='b', alpha=.3)
            ax[j].hist(YY[k], bins=bins, density=True, label=r'$\rho^N(X(T))$', color='r', alpha=.3)
            ax[j].plot(X, u0, label='$c(x,0)$', color='b')
            ax[j].plot(X, u, label='$c(x,T)$', color='r')
            ax[j].set(title=r'$W_1(\rho^N(T), \rho(T)) =$' + '{:3.2f}'.format(W[k]))
        fig.tight_layout()
        fig.subplots_adjust(top=0.935)
        plt.show()
        
        # filename = "Figures/Sample_" + f'{T=}'.split('=')[0] + str(T) + '_' + f'{a=}'.split('=')[0] + str(a) + '_' + f'{b=}'.split('=')[0] + str(b) + '_' + f'{c=}'.split('=')[0] + str(c) + '_' + 'σ' + str(sigma) + ".png"

    
#%% Sigma sweep

elif mode == 's':
    # parameters for potential
    l = 1/50
    C = .5
    r = .5
    
    Nsim = 3
    sigmas = np.logspace(-2, 0, Nsim)  # 0.2 # parameter of noise intensity
    fig, ax = plt.subplots(Nsim)
    fig.set_size_inches(10, 20, forward=True)
    # title = '$T = $' + str(T) + ', $a = $' + str(a) + ', $c = $' + str(c)
    # fig.suptitle(title, fontsize='x-large')
    
    for count, sigma in enumerate(sigmas):
        dt = np.float16((dx ** 2) / (2 * sigma ** 2))   # time step for Euler-Maruyama and finite
                                                    # difference method, chosen according the
                                                    # Von Neumann stability condition: k * dt / dx**2 <= 1/2
        NT = int(T / dt)  # number of time steps
        
        u0 = IC(L, dx, N, distr, modeBC, 'd')
        Y0 = IC(L, dx, N, distr, modeBC, 'r')
        
        start_time = time.time()
        u = finite_differences(NT, dt, X, nabla_F, deepcopy(u0), modeBC)
        Y = Euler_Maruyama(NT, dt, L, nabla_F, deepcopy(Y0), modeBC, sigma)
        print(str(count+1) + ' --- {:4.2f} s ---'.format(time.time() - start_time))
        
        # plot of Euler-Maruyama + finite-difference simulation at t=0 and t=T
        ax[count].hist(Y0, bins=bins, density=True, label=r'$\rho^N(X(0))$', color='b', alpha=.3)
        ax[count].hist(Y, bins=bins, density=True, label=r'$\rho^N(X(T))$', color='r', alpha=.3)
        
        ax[count].plot(X, u0, label='$c(x,0)$', color='b')
        ax[count].plot(X, u, label='$c(x,T)$', color='r')
        ax[count].set(xlabel='x', title='$\sigma =$' + str(sigma))
        ax[count].legend()
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.935)
    plt.show()  
    
    # filename = "Figures/Empirical_measure_" + f'{T=}'.split('=')[0] + str(T) + '_' + f'{a=}'.split('=')[0] + str(a) + '_' + f'{c=}'.split('=')[0] + str(c) + ".png"


#%% Parameter-'C' sweep

elif mode == 'r':
    sigma = .1
    # parameters for potential
    l = 1/50
    r = .5
    
    dt = np.float16((dx ** 2) / (2 * sigma ** 2))   # time step for Euler-Maruyama and finite
                                                # difference method, chosen according the
                                                # Von Neumann stability condition: k * dt / dx**2 <= 1/2
    NT = int(T / dt)  # number of time steps
    h = sigma*np.sqrt(dt)  # scaled time step for Euler-Maruyama = dx/np.sqrt(2)
    
    Nsim = 3
    Cs = np.linspace(0, .83, Nsim)  # repulsion intensity
    fig, ax = plt.subplots(Nsim)
    fig.set_size_inches(10, 20, forward=True)
    # title = '$T = $' + str(T) + ', $a = $' + str(a) + ', $b = $' + str(b) + ', $\sigma = $' + str(sigma)
    # fig.suptitle(title, fontsize='x-large')
    
    for count, C in enumerate(Cs):
        u0 = IC(L, dx, N, distr, modeBC, 'd')
        Y0 = IC(L, dx, N, distr, modeBC, 'r')
        
        start_time = time.time()
        u = finite_differences(NT, dt, X, nabla_F, deepcopy(u0), modeBC)
        Y = Euler_Maruyama(NT, dt, L, nabla_F, deepcopy(Y0), modeBC, sigma)
        print(str(count+1) + ' --- {:4.2f} s ---'.format(time.time() - start_time))
        
        # plot of Euler-Maruyama + finite-difference simulation at t=0 and t=T
        ax[count].hist(Y0, bins=bins, density=True, label=r'$\rho^N(X(0))$', color='b', alpha=.3)
        ax[count].hist(Y, bins=bins, density=True, label=r'$\rho^N(X(T))$', color='r', alpha=.3)
        
        ax[count].plot(X, u0, label='$c(x,0)$', color='b')
        ax[count].plot(X, u, label='$c(x,T)$', color='r')
        # ax[count].set(xlabel='x', title='$c =$' + str(c))
        ax[count].legend()
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.935)
    plt.show()  
    
    # filename = "Figures/Empirical_measure_" + distr + '_' + f'{T=}'.split('=')[0] + str(T) + '_' + f'{a=}'.split('=')[0] + str(a) + '_' + f'{b=}'.split('=')[0] + '_' + str(b) + '_' + f'{c=}'.split('=')[0] + str(c) + '_' + 'σ' + '_' + str(sigma) + ".png"
    

# if save_fig:
#     fig.savefig(filename, format="png")
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:26:42 2023

@author: Nathalie, Alberto and Steffi
"""

import numpy as np
from scipy.stats import binom, differential_entropy, entropy, norm, uniform, vonmises, vonmises_line

tol = 1e-5

#%% Position function

def position(x, L, modeBC):
    if modeBC == 't':
        return (x + L/2) % L - L/2
    else:
        return x


#%% Clustering reaction coordinate

def clustering_distance(Y, distance):
    N = len(Y)
    return 2/N/(N-1) * sum([1/distance(Y[i], Y[j]) for j in range(N) for i in range(N) if i != j])

# def clustering_distance(Y, distance, L, modeBC):
#     N = len(Y)
#     return 2/N/(N-1) * sum([sum(abs(position(Y[i] - Y[np.arange(N)!=i], L, modeBC))) for i in range(N)])


#%% Wasserstein distance

def Wasserstein(X1, X2, rho1, rho2, L, modeBC):
    if modeBC == 't':
        from ot import wasserstein_circle
        return wasserstein_circle(X1/L+.5, X2/L+.5, rho1, rho2)[0]
    else:
        from ot import wasserstein_1d
        return wasserstein_1d(X1, X2, rho1, rho2)


#%% Morse potential and force functions (three parameters)

def Morse_potential(x, L, l, C, r, modeBC):
    '''
    :param x: real number or vector
    :return: real number or vector
    '''
    if modeBC == 't':
        x = position(x, L, modeBC)
        return - np.cosh((np.abs(x)/L - .5)/l) / np.sinh(.5/l) + C * np.cosh((np.abs(x)/L - .5)/l/r) / np.sinh(.5/l/r)
    else:
        return - np.exp(-np.abs(x)/L/l) + C * np.exp(-np.abs(x)/L/l/r)

def Morse_force(x, L, l, C, r, modeBC):
    '''
    :param x: real number or vector
    :return: real number or vector
    '''
    if modeBC == 't':
        x = position(x, L, modeBC)
        return np.where(np.abs(x) < tol, 0, (- 1/L/l * np.sinh((np.abs(x)/L - .5)/l) / np.sinh(.5/l) + C/L/l/r * np.sinh((np.abs(x)/L - .5)/l/r) / np.sinh(.5/l/r)) * np.sign(x))
    else:
        return np.where(np.abs(x) < tol, 0, (-1/L/l * np.exp(-np.abs(x)/L/l) + C/L/l/r * np.exp(-np.abs(x)/L/l/r)) * np.sign(x))


#%% Kuramoto potential

def Kuramoto_potential(x, K, L, n):
    return - K * L**(-1/2) * np.cos(2*np.pi*n/L*x)

def Kuramoto_force(x, K, L, n):
    return K * 2*np.pi*n* L**(-3/2) * np.sin(2*np.pi*n/L*x)


#%% Morse potential and force functions (three different parameters)

def potential3(x, a, b, c):
    '''
    :param x: real number or vector
    :return: real number or vector
    '''
    # return - a * np.exp(- b * np.absolute(x)) + r * np.exp(- c * np.absolute(x))
    return a * (1 - np.exp(- b * (np.absolute(x) - c))) ** 2

def force3(x, a, b, c):
    '''
    :param x: real number or vector
    :return: real number or vector
    '''
    # return 2 * a * b * np.sign(x) * (np.exp(- b * (np.absolute(x) - c)) - np.exp(- 2 * b * (np.absolute(x) - c)))
    return np.where(np.abs(x) < tol, 0, 2 * a * b * np.sign(x) * (np.exp(- b * (np.absolute(x) - c)) - np.exp(- 2 * b * (np.absolute(x) - c))))


#%% Energy

# The entropy estimate in the deterministic case uses the built-in function 'entropy'
def entropy2(rho, N, modeBC):
    # if modeBC == 't':
    #     return np.sum(rho * np.log(rho)) * dx - np.log(uniform(-L/2, L).pdf(0))
    # else:
    #     return np.trapz(rho * np.log(rho), dx=dx) - np.log(uniform(-L/2, L).pdf(0))
    return - entropy(rho) + np.log(N)

# The entropy estimate in the stochastic case uses the built-in function 'differential_entropy'
def entropy2_s(Y, L):
    return - differential_entropy(Y) - np.log(uniform(-L/2, L).pdf(0))

# The interaction energy in the deterministic case is computed by numerical discretisation of the integrals
def interaction_energy(rho, L, dx, potential, modeBC):
    X = np.arange(-L/2, L/2, dx)
    if modeBC == 't':
        return np.sum([np.sum(np.roll(np.flip(potential(X)), len(X)//2+1+j) * rho) * dx for j, _ in enumerate(X)] * rho) * dx / 2
    else:
        return np.trapz([np.trapz(np.roll(np.flip(potential(np.arange(-L, L, dx))), len(X)+1+j)[:len(X)] * rho, dx=dx) for j, _ in enumerate(X)] * rho, dx=dx) / 2

# The interaction energy in the stochastic case is computed by using the empirical measure
def interaction_energy_s(Y, potential):
    N = len(Y)
    return sum([sum(potential(Y[i] - Y[np.arange(N)!=i])) for i in range(N)]) / 2 / N**2

# Deterministic energy
def energy(rho, L, dx, potential, sigma, modeBC):
    return sigma**2 * entropy2(rho, int(L/dx), modeBC) + interaction_energy(rho, L, dx, potential, modeBC)

# Stochastic energy
def energy_s(Y, L, potential, sigma):
    return sigma**2 * entropy2_s(Y, L) + interaction_energy_s(Y, potential)


#%% Random initial condition
# 'u': uniform; 'n': unimodal; 'b': bimodal; 'b2': closer bimodal; 'up': perturbed uniform

def IC(L, dx, N, distr, modeBC, mode):
    X = np.arange(-L/2, L/2, dx)
    m = L/4     # mean of initial distribution
    m2 = m/2    # other mean of initial distribution
    v = L/16    # width of initial distribution
    
    if distr == 'u':
        if mode == 'r':
            Y = uniform(-L/2, L).rvs(N)
        else:
            u = uniform(-L/2, L).pdf(X)
    elif distr == 'up':
        p = .99
        if modeBC == 'e':
            if mode == 'r':
                B = binom(N, p).rvs()
                Y = np.concatenate([uniform(-L/2, L).rvs(B), norm(0, v).rvs(N-B)])
            else:
                u = p*uniform(-L/2, L).pdf(X) + (1-p)*norm(0, v).pdf(X)
        elif modeBC == 't':
            if mode == 'r':
                B = binom(N, p).rvs()
                Y = np.concatenate([uniform(-L/2, L).rvs(B), vonmises(.5/v, scale=L/2/np.pi).rvs(N-B)])
            else:
                u  = p*uniform(-L/2, L).pdf(X) + (1-p)*vonmises(.5/v, scale=L/2/np.pi).pdf(X)
    elif distr == 'n':
        if modeBC == 'e':
            if mode == 'r':
                Y  = norm(0, v).rvs(N)
            else:
                u  = norm(0, v).pdf(X)
        elif modeBC == 't':
            if mode == 'r':
                Y  = vonmises(.5/v, scale=L/2/np.pi).rvs(N)
            else:
                u  = vonmises(.5/v, scale=L/2/np.pi).pdf(X)
    elif distr == 'b':
        if modeBC == 'e':
            if mode == 'r':
                B = binom(N, .5).rvs()
                Y = np.concatenate([norm(-m, v).rvs(B), norm(m, v).rvs(N-B)])
            else:
                u = norm(-m, v).pdf(X) / 2 + norm(m, v).pdf(X) / 2
        elif modeBC == 't':
            if mode == 'r':
                B = binom(N, .5).rvs()
                Y = np.concatenate([vonmises(.5/v, loc=-m, scale=L/4/np.pi).rvs(B), vonmises(.5/v, loc=m, scale=L/4/np.pi).rvs(N-B)])
            else:
                u = vonmises(.5/v, loc=m, scale=L/4/np.pi).pdf(X) / 2
    elif distr == 'b2':
        if modeBC == 'e':
            if mode == 'r':
                B = binom(N, .5).rvs()
                Y = np.concatenate([norm(-m2, v).rvs(B), norm(m2, v).rvs(N-B)])
            else:
                u = norm(-m2, v).pdf(X) / 2 + norm(m2, v).pdf(X) / 2
        elif modeBC == 't':
            if mode == 'r':
                B = binom(N, .5).rvs()
                Y = np.concatenate([vonmises(.5/v, loc=-m2, scale=L/4/np.pi).rvs(B), vonmises(.5/v, loc=m2, scale=L/4/np.pi).rvs(N-B)])
            else:
                u = vonmises_line(.5/v, loc=-m2, scale=L/4/np.pi).pdf(X) / 2 + vonmises_line(.5/v, loc=m2, scale=L/4/np.pi).pdf(X) / 2
    if mode == 'r':
        return Y
    else:
        return u


#%% PDE finite-difference solution

def finite_differences(NT, dt, X, force, u, modeBC, freq=10, k=1):
    # j = 1
    for t in range(NT - 1):
        # # display the advancement
        # if t > j*NT/freq:
        #     print(str(j) + '/' + str(freq))
        #     j += 1
        
        if modeBC == 't':
            # compute convolution between evaluated force and u (from time step before)
            C = np.array([np.sum(np.roll(np.flip(force(X)), len(X)//2+1+i) * u) for i, _ in enumerate(X)]) # we choose dx=1 and avoid dividing by dx later
            # update u
            u = u + (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / 4 \
                + dt * (np.roll(C, -1) * np.roll(u, -1) - np.roll(C, 1) * np.roll(u, 1)) / 2
        else:
            # compute convolution between evaluated force and u (from time step before)
            C = np.array([np.trapz(np.roll(np.flip(force(np.concatenate((X-X[-1], (X-X[0])[1:])))), len(X)+i)[:len(X)] * u) for i, _ in enumerate(X)]) # we choose dx=1 and avoid dividing by dx later
            # update u
            u[1:-1] = u[1:-1] + (np.roll(u, -1)[1:-1] - 2 * u[1:-1] + np.roll(u, 1)[1:-1]) / 4 \
                + dt * (np.roll(C, -1)[1:-1] * np.roll(u, -1)[1:-1] - np.roll(C, 1)[1:-1] * np.roll(u, 1)[1:-1]) / 2
            # no-flux boundary conditions (AM: why should they be 'no-flux'?)
            u[0] = u[1]
            u[-1] = u[-2]
        
    return u


#%% Simulation function for particle-based model

# Euler-Maruyama + finite-difference simulation run
def Euler_Maruyama(NT, dt, L, force, Y, modeBC, sigma, freq=10, k=1):
    h = sigma*np.sqrt(dt)  # scaled time step for Euler-Maruyama = dx/np.sqrt(2)
    N = len(Y)
    
    # j = 1
    for t in range(NT):
        # # display the advancement
        # if t > j*NT/freq:
        #     print(str(j) + '/' + str(freq))
        #     j += 1

        # Euler-Maruyama
        Y = position(Y - np.array([sum(force(Y[i] - Y[np.arange(N)!=i])) for i in range(N)]) / N * dt + np.random.normal(0, h, N), L, modeBC)
        
    return Y


#%% Simulation function for invariant measure

# Euler-Maruyama
def path(NT, Nsample, dt, L, h, force, Y0, modeBC, freq):
    N = len(Y0)
    Y = np.vstack((Y0, Y0 + np.random.normal(0, h, N), np.empty((NT-1, N), dtype=float)))
    # s = np.empty_like(Y0)
    
    j = 1
    for t in range(1, NT):
        # display the advancement
        if t > j*NT/freq:
            print(str(j) + '/' + str(freq))
            j += 1
        
        # Euler-Maruyama
        k = 0
        s = np.zeros(N)
        while t-k >= 0 and k <= Nsample:
            s = s + np.array([sum(force(Y[t][i] - Y[t-k][np.arange(len(Y[t-k]))!=i])) for i in range(N)])
            k += 1
        Y[t+1] = position(Y[t] - s / k / N * dt + np.random.normal(0, h, N), L, modeBC)
        
    return Y
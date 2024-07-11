## Importing standards libraries

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

## Importing libraries for the Gaussian Mixture Model

from scipy.stats import multivariate_normal
from scipy.stats import norm

## Definition of useful functions for the Gaussian Mixture Model

## Function to plot the density of a Gaussian Mixture Model
def generate_multimodal(centers, covariances, weights, plot_sample=None): 

    n_components = len(weights)

    if n_components != len(centers) or n_components != len(covariances):
        raise ValueError('The number of centers, covariances and weights should be the same')
    
    graph_len = max(max(abs(centers[:, 0])),  max(abs(centers[:, 1]))) + 5

    x_axis = np.linspace(-graph_len, graph_len, 500)
    y_axis = np.linspace(-graph_len, graph_len, 500)

    xx_axis, yy_axis = np.meshgrid(x_axis, y_axis)
    pos = np.dstack((xx_axis, yy_axis))
    
    # Compute the pdf for each center and sum them

    pdf = np.zeros(xx_axis.shape)

    for i in range(n_components):
        center = centers[i]
        covariance = covariances[i]
        weight = weights[i]

        rv = multivariate_normal(center, covariance) # On fixe la normale
        pdf += weight * rv.pdf(pos) # On évalue la densité de cette loi sur la grille des points et on somme (pondéremment)

    # Plot the result

    plt.figure(figsize=(8, 8))
    plt.contourf(xx_axis, yy_axis, pdf, levels=50, cmap='Blues', alpha=0.2) 
    ## levels : nombre de niveaux de couleurs ; cmap : choix de la palette de couleurs ; alpha : transparence
    plt.colorbar()
    
    if plot_sample is not None:
        x = plot_sample[:, 0]
        y = plot_sample[:, 1]
        plt.scatter(x, y, alpha=0.5, color = 'red')

    plt.title('Mixture of Gaussian Distributions density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


## Function to evaluate a point in the Gaussian Mixture Model

def evaluate_gaussian_mixture(x, weights, centers, covariances): 
    
    n_components = len(weights)

    dim_var = x.shape[0]

    pdf = np.zeros(1)

    for i in range(n_components):
        center = centers[i]
        covariance = covariances[i]
        weight = weights[i]

        if dim_var == 1:

            rv = norm(center, covariance)
            pdf += weight * rv.pdf(x)

        else : 

            rv = multivariate_normal(center, covariance) # On fixe la normale
            pdf += weight * rv.pdf(x) # On évalue la densité de cette loi sur la grille des points et on somme (pondéremment)

    return pdf[0]

## Compute the score vector of a Gaussian Mixture Model

def grad_multimodal(x, weights, centers, covariances): 
    
    n_components = len(weights)

    try : 

        dim_var = x.shape[0]

    except IndexError:

        dim_var = 1

    if len(centers) != n_components or len(covariances) != n_components:
        raise ValueError("The number of weights, means and covariances should be the same")

    gradient = np.zeros(x.shape)
    
    for i in range(n_components):
        center = centers[i]
        covariance = covariances[i]
        weight = weights[i]

        if dim_var == 1:

            rv = norm(center, covariance)
            pdf = rv.pdf(x)

            cov_inv = 1 / covariance
            diff = (x - center)

            gradient += weight * (cov_inv * diff) * pdf

        else : 
            
            rv = multivariate_normal(center, covariance)
            pdf = rv.pdf(x)
            
            cov_inv = np.linalg.inv(covariance)
            diff = x - center

            gradient += weight * (cov_inv @ diff) * pdf
        
    return gradient / evaluate_gaussian_mixture(x, weights, centers, covariances)

## Function to simulate the ULA algorithm

def ULA_true(x_init, nb_iter, step, weights, centers, covariances): 
    
    # Number of components of the Gaussian Mixture
    n_components = len(weights)

    try : 

        dim_var = x_init.shape[0]

    except IndexError:

        dim_var = 1

    x_tab = []

    x = x_init # Initialisation with a 2-dimensional standerd normal distribution

    for i in tqdm(range(nb_iter)):

        # Each iteration we compute the gradient of the target distribution and update the position of the particle
        grad = grad_multimodal(x, weights, centers, covariances) 

        #x = x + step * grad + np.sqrt(2 * step) * np.random.normal(size=dim_var)
        x = x - step * grad + np.sqrt(2 * step) * np.random.normal(size=dim_var)

        x_tab.append(x)
    
    return x_tab



## Function to plot the results of the ULA

def plot_ULA(nb_particles, step, nb_iter, weights, centers, covariances): 

    ## Initialisation : for the moment manually

    sample_init = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], nb_particles)

    x_result = []

    for j in tqdm(range(nb_particles)): 

        x_init = sample_init[j, :]

        traj = ULA_true(x_init, nb_iter, step, weights, centers, covariances)

        x_result.append(traj[-1])

    generate_multimodal(centers, covariances, weights, np.array(x_result))

    
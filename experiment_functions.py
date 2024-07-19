## Importing standards libraries

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

## Importing libraries for the Gaussian Mixture Model

from scipy.stats import multivariate_normal

## Convergence metrics

from scipy.stats import wasserstein_distance
from scipy.stats import entropy # KL-divergence


## Definition of useful functions for the Gaussian Mixture Model

##First, this function plots the density of the target distribution (Gaussian Mixture) and the praticles given as an argument

def generate_multimodal(centers, covariances, weights, plot_sample=None, grad_sample = None): 

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
    plt.contourf(xx_axis, yy_axis, pdf, levels=50, cmap='Blues', alpha=0.4) 
    ## levels : nombre de niveaux de couleurs ; cmap : choix de la palette de couleurs ; alpha : transparence
    plt.colorbar()
    
    if plot_sample is not None:
        x = plot_sample[:, 0]
        y = plot_sample[:, 1]
        plt.scatter(x, y, alpha=0.5, s = 8, color = 'red')

    if grad_sample is not None : 
        grad_x = grad_sample[:, 0]
        grad_y = grad_sample[:, 1]
        print(grad_x, grad_y)
        plt.quiver(x, y, grad_x, grad_y)

    plt.title('Mixture of Gaussian Distributions density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


## This function is to evaluate a vector of points in a Gaussian Mixture density 

def evaluate_gaussian_mixture(x, weights, centers, covariances): 

    n_components = len(weights)

    pdf = np.zeros(x.shape[0]) #Question

    for i in range(n_components):
        center = centers[i]
        weight = weights[i]
        covariance = covariances[i]

        rv = multivariate_normal(center, covariance)
        pdf = rv.pdf(x) 

        pdf_tot += pdf * weight

    return pdf_tot


## This function computes the gradient of the log-density of a Gaussian Mixture for a vector of points

def grad_multimodal_opti(x, weights, centers, covariances, gamma_t = 1): 
    
    n_components = len(weights)

    gradient = np.zeros(x.shape)

    covariance = covariances[0]

    cov_inv = np.linalg.inv(covariance)

    pdf_tot = np.zeros(x.shape[0])
    
    for i in range(n_components):
        center = centers[i]
        weight = weights[i]

        rv = multivariate_normal(center, covariance)
        pdf = rv.pdf(x) 

        pdf_tot += pdf * weight #Denominator term
            
        diff = x - center

        gradient += weight * (np.einsum('ij,kj->ik', cov_inv, diff).T * pdf[:, np.newaxis]) #Numerator term


    gradient = (-1) * (gradient / pdf_tot[:, np.newaxis])

    return gradient


#We define the Unadjusted Langevin Algorithm (constant step + non stochastic gradient)

def ULA(x_init, nb_iter, step, weights, centers, covariances): 

    x = x_init

    stochastic_term = np.zeros(x.shape[0])

    gradient_term = np.zeros(x.shape[0])

    for i in tqdm(range(nb_iter)):

        # Each iteration we compute the gradient of the target distribution and update the position of the particles
        grad = grad_multimodal_opti(x, weights, centers, covariances)

        noise =  np.sqrt(2 * step) * np.random.randn(1000, 2)

        stochastic_term += np.linalg.norm(noise, axis = 1)

        gradient_term += np.linalg.norm(step * grad, axis = 1) #Size of this vector is nb_particles

        x = x + step * grad + noise

    generate_multimodal(centers, covariances, weights, x)

    return f'The magnitude of the Stochastic term is {np.mean(stochastic_term / nb_iter)} whereas the magnitude of the gradient term is { np.mean(gradient_term / nb_iter)}'


## We define the function to implement the ULA with Dilation path 

def ULA_dilation(x_init, nb_iter, step, weights, centers, covariances, end_schedule,  bound = 100, alpha = 1): 

    x = x_init

    stochastic_term = np.zeros(x.shape[0])

    gradient_term = np.zeros(x.shape[0])

    time = np.zeros(x.shape[0]) #Each particle follows its own time-line ? ? 

    step_tab = np.ones(x.shape[0]) * step

    for i in tqdm(range(nb_iter)):

        time += step_tab

        schedule = np.minimum(end_schedule, time) / end_schedule #end_schedule is the date from which we are using the reel target distribution

        gamma = 1 / np.sqrt(schedule)

        # Each iteration we compute the gradient of the target distribution and update the position of the particle
        grad = gamma[:, np.newaxis] * grad_multimodal_opti(gamma[:, np.newaxis] * x, weights, centers, covariances)

        step_tab = np.minimum(1 / (np.linalg.norm(grad, axis = 1) + 1e-7), bound) * alpha #Vecteur de taille nb_particles qui donne le step pour chaque particle à cette itération

        noise =  np.sqrt(2 * step_tab)[:, np.newaxis] * np.random.randn(1000, 2) #bien terme à terme pour sqrt 
        
        stochastic_term += np.linalg.norm(noise, axis = 1)

        grad_update = step_tab[:, np.newaxis] * grad
        
        gradient_term += np.linalg.norm(grad_update, axis = 1) #Vecteur de taille nb_particles 

        x = x + grad_update + noise

    generate_multimodal(centers, covariances, weights, x)

    print(f'The magnitude of the Stochastic term is {np.mean(stochastic_term / nb_iter)} whereas the magnitude of the gradient term is { np.mean(gradient_term / nb_iter)}')

    return f' Voici le step moyen sur toutes les particles {np.mean(time / nb_iter)}, et voici le time moyen auquel on est sur la simu {np.mean(time)}'


## We define the function to implement the ULA with geometric path 

def ULA_geometric(x_init, weights, centers, covariances, step, nb_iter, end_schedule): 

    x = x_init
    
    stochastic_term = np.zeros(x.shape[0])

    gradient_term = np.zeros(x.shape[0])

    time = np.zeros(x.shape[0]) #Each particle follows its own time-line ? ? 

    step_tab = np.ones(x.shape[0]) * step

    for i in tqdm(range(nb_iter)):
        
        time += step

        schedule = np.minimum(end_schedule, time) / end_schedule

        # Each iteration we compute the gradient of the target distribution and update the position of the particle
        grad = schedule[:, np.newaxis] * grad_multimodal_opti(x, weights, centers, covariances, 1) + (1 - schedule)[:, np.newaxis] * x

        noise =  np.sqrt(2 * step_tab)[:, np.newaxis] * np.random.randn(1000, 2) #bien terme à terme pour sqrt 
        
        stochastic_term += np.linalg.norm(noise, axis = 1)

        grad_update = step_tab[:, np.newaxis] * grad
        
        gradient_term += np.linalg.norm(grad_update, axis = 1) #Vecteur de taille nb_particles 

        x = x + grad_update + noise

    
    generate_multimodal(centers, covariances, weights, x)

    print(f'The magnitude of the Stochastic term is {np.mean(stochastic_term / nb_iter)} whereas the magnitude of the gradient term is { np.mean(gradient_term / nb_iter)}')

    return f' Voici le step moyen sur toutes les particles {np.mean(time / nb_iter)}, et voici le time moyen auquel on est sur la simu {np.mean(time)}'


## Now, we define functions that are the convergence metrics used to measure the distance between the sample obtained and 
# the actual target distribution (To Be Tested)

## The is the Inverse Multiquadratic Kernel

def multiquad_kernel(x, y, beta = 0.5): 

    if len(x) != len(y): 
        raise ValueError('Given vectors have not the same dimension')
    
    norm = sum((x - y) ** 2)

    return (1 + norm) ** (-beta)


## To compute the partial derivatives of the Inverse Multiquadratic Kernel

def grad_multiquad_kern(x, y, beta=0.5):

    diff = x - y

    #Returns an array with the gradient w.r.t x and w.r.t y

    return [(-2*beta) * multiquad_kernel(x, y, beta + 1) * diff , (2*beta) * multiquad_kernel(x, y, beta + 1) * diff]


## Function to compute the second intermediary Kernel for the KSD

def compute_kernel(x, y, weights, centers, covariances, beta = 0.5): 

    dim_var = x.shape[0]

    nb_components = len(weights)

    if y.shape[0] != dim_var: 
        raise ValueError('Dimension problem : the two vectors have different dimensions')
    
    elif len(centers) != nb_components or len(covariances) != nb_components:
        raise ValueError('Dimension problem : parameters of the target distribution have different lenths')
    
    grad_vect_1 = grad_multimodal_opti(x, weights, centers, covariances)
    
    grad_vect_2 = grad_multimodal_opti(y, weights, centers, covariances)

    kernel = multiquad_kernel(x, y, beta)

    grad_ = grad_multiquad_kern(x, y, beta)

    grad_MK_x = grad_[0]

    grad_MK_y = grad_[1]

    return kernel * (grad_vect_1.T @ grad_vect_2) + (grad_vect_1.T @ grad_MK_y) + (grad_MK_x.T @ grad_vect_2) + (grad_MK_x.T @ grad_MK_y)


## Function to compute the (approximation of) Kernel Stein Discrepancy (To Be Tested)

def squared_KSD(intermediate_sample, weights, centers, covariances, beta = 0.5): 

    nb_components = len(weights)

    sample_size = len(intermediate_sample)

    sample_split_size = int(sample_size // 2)

    if len(centers) != nb_components or len(covariances) != nb_components: 
        raise ValueError('Dimension problem : parameters of the target distribution have different lenths')

    ## On scinde l'échantillon en deux échantillons distincts, issus de la distribution intermédiaire et dont chaque réalisation est indépendente
    if sample_size % 2 == 1:

        intermediate_sample = intermediate_sample[:-1]

    sample_1 = intermediate_sample[:-sample_split_size]
    sample_2 = intermediate_sample[sample_split_size:]

    estimator = 0

    for i in tqdm(range(sample_split_size)): 

        vect_1 = sample_1[i, :]
        vect_2 = sample_2[i, :]

        estimator += compute_kernel(vect_1, vect_2, weights, centers, covariances, beta)
    
    return estimator / sample_split_size


## Then, we use packages to implement the other distances : 

#First, the KL-Divergence

def KL_divergence(distrib_P, distrib_Q): 
    
    return entropy(pk=distrib_P, qk=distrib_Q)


#Then, Wasserstein Distance between two distributions

def wasser_dist(distrib_P, distrib_Q): 

    return wasserstein_distance(distrib_P, distrib_Q)


# Multimodality Score (the root mean squared error between the actual and expected number of particles per mode)

def MMS(intermediate_sample, weights, centers):

    nb_components = len(weights)

    sample_size = len(intermediate_sample)

    if len(centers) != nb_components: 
        raise ValueError('Dimension Problem')

    particle_by_mode = np.zeros(nb_components)

    for i in tqdm(range(sample_size)): 
        
        item = intermediate_sample[i]
        dist = []

        for j in range(nb_components): 

            center = centers[j]

            dist.append(sum((center - item) ** 2))

        mode = np.argmax(dist)

        particle_by_mode[mode] += 1

    return np.sqrt(np.mean((particle_by_mode - weights) ** 2))


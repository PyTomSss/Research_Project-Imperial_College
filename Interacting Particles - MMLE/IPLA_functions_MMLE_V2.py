## Importing standards libraries

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
#import torch

## Importing libraries for the Gaussian Mixture Model

from scipy.stats import multivariate_normal
from scipy.stats import norm
import seaborn as sns

## Convergence metrics

from scipy.stats import wasserstein_distance
from scipy.stats import entropy # KL-divergence


## Import functions for the experiment : 

from experiment_functions import *
from IPLA_Exp_Functions import *


##########################################################################################################################################################################
############### HERE WE DEFINE USEFUL FUNCTIONS RELATED TO THE EXPERIMENT : GENERATION OF PARAMETERS, SAMPLING FROM A GAUSSIAN MIXTURE, PLOT A SAMPLE....#################
######################################################################################################V#################V#################################################
 
def gen_prior_param(dx, size_cube): 
    """
    Function to generate the parameters of the prior distribution of the experiment
    """
    
    semi_size = size_cube // 2
    range_values = np.array([-2, -1, 0, 1, 2])  # Values for i and j
    grid = np.array(np.meshgrid(range_values, range_values)).T.reshape(-1, 2)  

    means = []
    for (i, j) in grid:
        mean_vector = np.tile([semi_size*i, semi_size*j], dx // 2)  # Create a pattern 
        means.append(mean_vector)

    means_prior = np.array(means)

    # Verify the shape
    print("Shape of means array:", means_prior.shape)

    ## Covariance
    covariances_prior = np.array([np.eye(dx)] * 25)
    print(f"Unit Covariance Matrix: {covariances_prior[0]}")

    # Weights
    weights_prior = np.ones(25) / 25
    print(f'Weights vector : {weights_prior}')

    return means_prior, covariances_prior, weights_prior



def sample_prior_dx(nb_particle_gen, centers_prior, covariances_prior, weights_prior) : 
    """
    Function to sample from a Gaussian Mixture given its parameters (means, covariances and weights) and the wanted size of the sample. 
    In the context of the experiment, we use it to initialize the algorithm and to manually generate the observation Y
    """

    dx = centers_prior[0].shape[0]
    nb_components = len(weights_prior)

    x_star = np.zeros((nb_particle_gen, dx))

    # Selection of the component randomly
    component_choices = np.random.choice(nb_components, size=nb_particle_gen, p=weights_prior) #choix aléatoire entre les normales de la mixture (pondéré)

    for i in range(nb_components):

        num_samples = np.sum(component_choices == i) #

        if num_samples > 0:

            #x_star[component_choices == i] = np.random.multivariate_normal(mean=centers[i], cov=covariances[i], size=num_samples) #pour les indices où l'on a tiré la composante en question on tire selon celle-ci
            x_star[component_choices == i] = np.random.multivariate_normal(mean=centers_prior[i], cov=covariances_prior[i], size=num_samples) 

    return x_star



def plot_sample_dx(sample_to_plot_1, desc_sample_1 = '', sample_to_plot_2=None, desc_sample_2 = ''): 

    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 10))
    plt.scatter(sample_to_plot_1[:, 0], sample_to_plot_1[:, 1], alpha=0.6, s=50, color="red", label = desc_sample_1, edgecolor='k')
    
    if sample_to_plot_2 is not None: 

        plt.scatter(sample_to_plot_2[:, 0], sample_to_plot_2[:, 1], alpha=0.05, s=50, color=sns.color_palette("husl", 8)[2], label = desc_sample_2, edgecolor='k')
    
    plt.title('Sample from a Gaussian Mixture Distribution', fontsize=16, weight='bold')
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_facecolor('#f7f7f7')
    
    plt.legend()
    plt.show()
 


def generate_obs_dx(x_star, theta, sigma_y) : 
     """
    
    Function that generates the observed data point as it is explained in the Appendix of the paper : y = Hx + z. 
    x is drawn according to the prior distribution and z is a gaussian noise whose parameters are defined in the Appendix. 

    """
    
     noise = np.random.normal(0, sigma_y**2)

     y_obs = np.dot(theta, x_star.T) + noise

     return y_obs



def post_params_dx(theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs, plot = False):
    """
    
    In the context of this experiment, the posterior distribution is also a Gaussian Mixture whose parameters can be analytically computed. 
    This function permits to generates those parameters given the parameters of the model and of the prior distribution.  

    """

    dx = theta.shape[0]

    nb_components = weights_prior.shape[0]

    theta_reshape = theta.reshape(1, dx) #To behave as a matrix

    Sigma_post = np.linalg.inv(np.eye(dx) + (1 / sigma_y**2) * np.outer(theta_reshape.T, theta_reshape))

    covariances_posteriori = np.full((nb_components, dx, dx), Sigma_post)

    weights_posteriori = np.zeros(nb_components)

    centers_posteriori = np.zeros((nb_components, dx))

    for i in range(nb_components): 
        
        M = (theta * (y_obs / (sigma_y ** 2)))
        
        centers_posteriori[i] = np.dot(Sigma_post, M + centers_prior[i]) ## WTF OUBLI M ? 

        ## Weights
        mean_pdf = theta @ centers_prior[i].T

        covariance_pdf = sigma_y**2 + np.dot(theta_reshape, theta_reshape.T)

        weights_posteriori[i] = weights_prior[i] * norm.pdf(y_obs, mean_pdf, covariance_pdf)

    weights_posteriori = weights_posteriori / np.sum(weights_posteriori)

    return centers_posteriori, covariances_posteriori, weights_posteriori



##########################################################################################################################################################################
########## HERE WE DEFINE DIFFERENTS ALGORITHMS TO SAMPLE FROM THE POSTERIOR DISTRIBUTION OF THE MODEL KNOWING THE PARAMETERS : ULA, ULA with Dilation Path... ###########
######################################################################################################V#################V#################################################

def grad_descent_post_dx(sample_init, step_size, nb_iter, centers_prior, covariances_prior, weights_prior, y_obs, sigma_y, true_theta, plot = True): 

    sample_size = sample_init.shape[0]
    dim_var = sample_init.shape[1]

    traj = np.zeros((nb_iter, sample_size, dim_var))

    centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
    grad = np.zeros((sample_size, dim_var))
    for i in tqdm(range(nb_iter)): 
        
        #grad = grad_multimodal_opti(sample_init, weights_post, centers_post, covariances_post)
        grad = ((1/sigma_y**2) * true_theta[:, np.newaxis] * (y_obs - np.dot(true_theta, sample_init.T))).T 

        grad += grad_multimodal_opti(sample_init, weights_prior, centers_prior, covariances_prior)
        #grad += np.nan_to_num(grad_multimodal_opti(sample_init, weights_prior, centers_prior, covariances_prior), nan = 0)

        sample_init += step_size * grad # + np.sqrt(step_size * 2) * np.random.randn(sample_size, dim_var)

        traj[i] = sample_init

    if plot : 

        centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
        
        sample_post = sample_prior_dx(1000, centers_post, covariances_post, weights_post)
        
        plot_sample_dx(sample_init, "ULA", sample_post, "True Posterior Sample") 

    return sample_init



def ULA_post_dx(sample_init, step_size, nb_iter, centers_prior, covariances_prior, weights_prior, y_obs, sigma_y, true_theta, plot = True): 

    sample_size = sample_init.shape[0]
    dim_var = sample_init.shape[1]

    traj = np.zeros((nb_iter, sample_size, dim_var))

    stochastic_term = np.zeros(nb_iter)

    gradient_term = np.zeros(nb_iter)

    for i in tqdm(range(nb_iter)): 
        
        #grad = grad_multimodal_opti(sample_init, weights_post, centers_post, covariances_post)
        grad = ((1/sigma_y**2) * true_theta[:, np.newaxis] * (y_obs - np.dot(true_theta, sample_init.T))).T 

        grad += grad_multimodal_opti(sample_init, weights_prior, centers_prior, covariances_prior)
        #grad += np.nan_to_num(grad_multimodal_opti(sample_init, weights_prior, centers_prior, covariances_prior), nan = 0)

        noise = np.sqrt(step_size * 2) * np.random.randn(sample_size, dim_var)

        sample_init += step_size * grad + noise

        stochastic_term[i] = np.nanmean(np.linalg.norm(noise, axis = 1))

        gradient_term[i] = np.nanmean(np.linalg.norm(step_size * grad, axis = 1)) #Size of this vector is nb_particles

        traj[i] = sample_init

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(nb_iter), stochastic_term, color = 'red', label = 'Stochatic Term')
    plt.plot(np.arange(nb_iter), gradient_term, color = 'green', label = 'Gradient Term')
    plt.legend()
    plt.show()

    if plot : 

        centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
        
        sample_post = sample_prior_dx(1000, centers_post, covariances_post, weights_post)
        
        plot_sample_dx(sample_init, "ULA", sample_post, "True Posterior Sample") 

    return sample_init



def ULA_dilation_exp(sample_init, step_size, nb_iter, centers_prior, covariances_prior, weights_prior, y_obs, sigma_y, true_theta, start_schedule,
                    end_schedule, plot = True): 
    
    sample_size = sample_init.shape[0]
    dim_var = sample_init.shape[1]

    traj = np.zeros((nb_iter, sample_size, dim_var))

    stochastic_term = np.zeros(nb_iter)

    gradient_term = np.zeros(nb_iter)

    time_SDE = 0

    for i in tqdm(range(nb_iter)): 

        time_SDE += step_size

        schedule = np.minimum(start_schedule + np.minimum(end_schedule, time_SDE) / end_schedule, 1)

        gamma = 1 / np.sqrt(schedule)

        gamma_sample = gamma * sample_init
        
        #grad = grad_multimodal_opti(sample_init, weights_test, centers_test, covariances_test)
        grad = gamma * ((1/sigma_y**2) * true_theta[:, np.newaxis] * (y_obs - np.dot(true_theta, gamma_sample.T))).T 

        grad += gamma * grad_multimodal_opti(gamma_sample, weights_prior, centers_prior, covariances_prior)

        taming_coef = step_size / (1 + step_size * np.linalg.norm(grad, axis = 1))

        grad_update = taming_coef[:, np.newaxis] * grad

        noise = np.sqrt(step_size * 2) * np.random.randn(sample_size, dim_var)

        sample_init += grad_update + noise

        stochastic_term[i] = np.nanmean(np.linalg.norm(noise, axis = 1))

        gradient_term[i] = np.nanmean(np.linalg.norm(grad_update, axis = 1)) #Size of this vector is nb_particles

        traj[i] = sample_init

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(nb_iter), stochastic_term, color = 'red', label = 'Stochatic Term')
    plt.plot(np.arange(nb_iter), gradient_term, color = 'green', label = 'Gradient Term')
    plt.legend()
    plt.show()
    
    if plot : 

        centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
        
        sample_post = sample_prior_dx(1000, centers_post, covariances_post, weights_post)
        
        plot_sample_dx(sample_init, "ULA with Dilation Path", sample_post, "True Posterior Sample") 

    return sample_init
    


def ULA_dilation_exp_adapt(sample_init, step_size, nb_iter, centers_prior, covariances_prior, weights_prior, y_obs, sigma_y, true_theta, start_schedule,
                    end_schedule, alpha = 1, bound = 100, plot = True): 
    
    sample_size = sample_init.shape[0]
    dim_var = sample_init.shape[1]

    traj = np.zeros((nb_iter, sample_size, dim_var))

    stochastic_term = np.zeros(nb_iter)

    gradient_term = np.zeros(nb_iter)

    time_SDE = np.zeros(sample_size) #Each particle follows its own time-line 

    step_tab = np.full(sample_size, start_schedule)  

    for i in tqdm(range(nb_iter)): 

        time_SDE += step_tab

        schedule = np.minimum(end_schedule, time_SDE) / end_schedule

        gamma = 1 / np.sqrt(schedule)

        gamma_sample = gamma[:, np.newaxis] * sample_init
        
        grad = ((1/sigma_y**2) * true_theta[:, np.newaxis] * (y_obs - np.dot(true_theta, gamma_sample.T))).T 

        grad += grad_multimodal_opti(gamma_sample, weights_prior, centers_prior, covariances_prior)

        grad = gamma[:, np.newaxis] * grad

        step_tab = np.minimum(1 / (np.linalg.norm(grad, axis = 1) + 1e-8), bound) * alpha #Vecteur de taille nb_particles qui donne le step pour chaque particle à cette itération

        noise = np.sqrt(2 * step_tab)[:, np.newaxis] * np.random.randn(sample_size, dim_var)

        grad_update = step_tab[:, np.newaxis] * grad

        sample_init += grad_update + noise

        stochastic_term[i] = np.nanmean(np.linalg.norm(noise, axis = 1))

        gradient_term[i] = np.nanmean(np.linalg.norm(grad_update, axis = 1)) #Size of this vector is nb_particles

        traj[i] = sample_init

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(nb_iter), stochastic_term, color = 'red', label = 'Stochatic Term')
    plt.plot(np.arange(nb_iter), gradient_term, color = 'green', label = 'Gradient Term')
    plt.legend()
    plt.show()
    
    if plot : 

        centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
        
        sample_post = sample_prior_dx(1000, centers_post, covariances_post, weights_post)
        
        plot_sample_dx(sample_init, "ULA with Dilation Path", sample_post, "True Posterior Sample") 

    return sample_init
 


###NEED TO BE MODIFIED
def ULA_dilation_exp_adapt_precision(sample_init, step_size, nb_iter, centers_prior, covariances_prior, weights_prior, y_obs, sigma_y, true_theta, start_schedule,
                    end_schedule, alpha = 1, bound = 100, plot = True): 
    
    sample_size = sample_init.shape[0]
    dim_var = sample_init.shape[1]

    traj = np.zeros((nb_iter, sample_size, dim_var))

    time_SDE = np.zeros(sample_size) #Each particle follows its own time-line 

    step_tab = np.full(sample_size, start_schedule)  

    for i in tqdm(range(nb_iter)): 

        time_SDE += step_tab

        schedule = np.minimum(end_schedule, time_SDE) / end_schedule

        gamma = 1 / np.sqrt(schedule)

        gamma_sample = gamma[:, np.newaxis] * sample_init
        
        grad = ((1/sigma_y**2) * true_theta[:, np.newaxis] * (y_obs - np.dot(true_theta, gamma_sample.T))).T 

        grad += grad_multimodal_opti_precision(gamma_sample, weights_prior, centers_prior, covariances_prior)

        grad = gamma[:, np.newaxis] * grad

        step_tab = np.minimum(1 / (np.linalg.norm(grad, axis = 1) + 1e-8), bound) * alpha #Vecteur de taille nb_particles qui donne le step pour chaque particle à cette itération

        noise = np.sqrt(2 * step_tab)[:, np.newaxis] * np.random.randn(sample_size, dim_var)

        grad_update = step_tab[:, np.newaxis] * grad

        sample_init += grad_update + noise

        traj[i] = sample_init

    
    if plot : 

        centers_post, covariances_post, weights_post = post_params_dx(true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)
        
        sample_post = sample_prior_dx(1000, centers_post, covariances_post, weights_post)
        
        plot_sample_dx(sample_init, "ULA with Dilation Path", sample_post, "True Posterior Sample") 

    return sample_init
    
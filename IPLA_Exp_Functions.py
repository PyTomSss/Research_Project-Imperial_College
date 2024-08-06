## Importing standards libraries

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

## Importing libraries for the Gaussian Mixture Model

from scipy.stats import multivariate_normal
from scipy.stats import norm

## Autograd packages

## Convergence metrics

from scipy.stats import wasserstein_distance
from scipy.stats import entropy # KL-divergence
import ot #Optimal Transport


## Import functions for the experiment : 

from experiment_functions import *


def sample_prior(nb_particle_gen, centers_prior, covariances_prior, weights_prior) : 
    """

    Function to sample from a Gaussian Mixture given its parameters (means, covariances and weights) and the wanted size of the sample. 
    In the context of the experiment, we use it to initialize the algorithm and to manually generate the observation Y

    """

    nb_components = len(weights_prior)
    x_star = np.zeros((nb_particle_gen, 2))

    # Selection of the component randomly
    component_choices = np.random.choice(nb_components, size=nb_particle_gen, p=weights_prior) #choix aléatoire entre les normales de la mixture (pondéré)

    for i in range(nb_components):

        num_samples = np.sum(component_choices == i) #

        if num_samples > 0:

            #x_star[component_choices == i] = np.random.multivariate_normal(mean=centers[i], cov=covariances[i], size=num_samples) #pour les indices où l'on a tiré la composante en question on tire selon celle-ci
            x_star[component_choices == i] = np.random.multivariate_normal(mean=centers_prior[i], cov=covariances_prior[i], size=num_samples) 

    return x_star


def generate_obs(x_star, theta, sigma_y) : 
    """
    
    Function that generates the observed data point as it is explained in the Appendix of the paper : y = Hx + z. 
    x is drawn according to the prior distribution and z is a gaussian noise whose parameters are defined in the Appendix. 

    """

    noise = np.random.normal(0, sigma_y**2)

    y_obs = np.dot(theta, x_star.T) + noise

    return y_obs


def post_params(theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs, plot = False):
    """
    
    In the context of this experiment, the posterior distribution is also a Gaussian Mixture whose parameters can be analytically computed. 
    This function permits to generates those parameters given the parameters of the model and of the prior distribution.  

    """

    dx = theta.shape[0]

    nb_components = weights_prior.shape[0]

    theta_reshape = theta.reshape(1, 2) #To behave as a matrix

    Sigma_post = np.linalg.inv(np.eye(dx) + (1 / sigma_y**2) * np.outer(theta_reshape.T, theta_reshape))

    covariances_posteriori = np.full((nb_components, dx, dx), Sigma_post)

    weights_posteriori = np.zeros(nb_components)

    centers_posteriori = np.zeros((nb_components, dx))

    for i in range(nb_components): 
        
        M = (theta * (y_obs / (sigma_y ** 2)))
        
        centers_posteriori[i] = np.dot(Sigma_post, M + centers_prior[i]) ## WTF OUBLI M ? 

        ## Weights
        mean = theta @ centers_prior[i].T

        covariance = sigma_y**2 + np.dot(theta_reshape, theta_reshape.T)
    
        weights_posteriori[i] = weights_prior[i] * norm.pdf(y_obs, mean, covariance)

    weights_posteriori = weights_posteriori / np.sum(weights_posteriori)

    if plot : 

        generate_multimodal(centers_posteriori, covariances_posteriori, weights_posteriori)

    return centers_posteriori, covariances_posteriori, weights_posteriori


def PGD(nb_particles, nb_iter, step_size, centers_prior, covariances_prior, weights_prior, theta_0, sigma_y, y_obs,
        coef_particle = 1, plot = False, plot_true_theta = None) : 
    """
    This function executes the Particle Gradient Descent in the context of our experiment. Given :
    - The number of particles
    - The number of iterations
    - The step size
    - Parameters of the prior distribution
    - The observed data point "y"
    - The initializing theta_0
    """

    theta_t = theta_0

    dx = theta_0.shape[0]

    sample = sample_prior(nb_particles, centers_prior, covariances_prior, weights_prior)

    time_SDE = 0

    theta_traj = np.zeros((nb_iter, dx))

    for i in tqdm(range(nb_iter)) : 

        #We don't need to compute the parameters of the posteriori distribution given the updated theta because we use another formula
        #centers_post, covariances_post, weights_post = post_params(theta_t, sigma_y, centers_prior, covariances_prior, weights_prior)

        time_SDE += step_size

        ## on prend le gradient selon x de la posterior actualisée avec theta_t et qui est aussi une mixture Gaussienne
        grad = ((1/sigma_y**2) * theta_t[:, np.newaxis] * (y_obs - np.dot(theta_t, sample.T))).T 

        grad += grad_multimodal_opti(sample, weights_prior, centers_prior, covariances_prior) 

        grad_update = step_size * grad * coef_particle
        
        #Noise
        noise =  np.sqrt(2 * step_size) * np.random.randn(nb_particles, dx)

        sample += grad_update + noise #Warning sign

        #MAJ THETA we need a fct that compute gradient of the potential wrt to theta

        grad_theta = grad_theta_GM(sample, theta_t, y_obs, sigma_y) #renvoie un vecteur avec tous les gradients

        grad_theta_update = np.sum(grad_theta, axis = 0) ## ON MULTIPLIE CHAQUE PARTICLE A SON PAS SPECIFIQUE

        theta_t = theta_t - (step_size / nb_particles) * grad_theta_update 

        theta_traj[i] = theta_t

    centers_post, covariances_post, weights_post = post_params(plot_true_theta, sigma_y, centers_prior, covariances_prior, weights_prior, y_obs)

    if plot :

        generate_multimodal(centers_post, covariances_post, weights_post, sample)

        plt.plot(np.arange(nb_iter), theta_traj)
        plt.show()

    return theta_traj


def marginal_likelihood_obs(theta, x_samples, y, sigma_y):
    """
    This function computes the log-likelihood of a parameter theta in the context of our experiment, given the observed data point "y"
    """

    likelihoods = []

    for x in x_samples:

        mean = np.dot(theta, x)

        ##LOG-LIKELIHOOD
        likelihood = multivariate_normal.logpdf(y, mean=mean, cov=sigma_y**2)
        likelihoods.append(likelihood)
        
    return np.mean(likelihoods)
    
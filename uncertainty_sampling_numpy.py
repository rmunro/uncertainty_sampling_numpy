"""NumPy Uncertainty Sampling examples

The four types of Uncertainty Sampling in this repository are:

Least Confidence: difference between the most confident prediction and 100% confidence

Margin of Confidence: difference between the top two most confident predictions

Ratio of Confidence: ratio between the top two most confident predictions

Entropy: difference between all predictions, as defined by information theory

"""

import numpy as np
import scipy.stats


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"


def margin_confidence(prob_dist, sorted=False):
	""" Margin of Confidence Uncertainty Sampling

	Returns the uncertainty score of a probability distribution using
	margin of confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if not sorted:
		prob_dist[::-1].sort() # sort probs so that largest is at prob_dist[0]		
		
	difference = (prob_dist[0] - prob_dist[1])
	margin_conf = 1 - difference 
	
	return margin_conf
	

def ratio_confidence(prob_dist, sorted=False):
	"""Ratio of Confidence Uncertainty Sampling 
 
	Returns the uncertainty score of a probability distribution using
	ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if not sorted:
		prob_dist[::-1].sort() # sort probs so that largest is at prob_dist[0]		
		
	ratio_conf = prob_dist[1] / prob_dist[0]
	
	return ratio_conf
	
	


def least_confidence(prob_dist, sorted=False):
	""" Least Confidence Uncertainty Sampling 

	Returns the uncertainty score of a probability distribution using
	least confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if sorted:
		simple_least_conf = prob_dist[0] # most confident prediction
	else:
		simple_least_conf = np.nanmax(prob_dist) # most confident prediction, ignoring NaNs
				
	num_labels = float(prob_dist.size) # number of labels
	
	normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
	
	return normalized_least_conf



def entropy_score(prob_dist):
	""" Entropy-Based Uncertainty Sampling 

	Returns the uncertainty score of a probability distribution using
	entropy score
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	log_probs = prob_dist * np.log2(prob_dist) # multiply each probability by its base 2 log
	raw_entropy = 0-np.sum(log_probs)

	normalized_entropy = raw_entropy / np.log2(prob_dist.size)
	
	return normalized_entropy
	
	

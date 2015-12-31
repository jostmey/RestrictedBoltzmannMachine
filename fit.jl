##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-12-23
# Environment: Julia v0.4
# Purpose: Fit bias terms to the features in the dataset.
##########################################################################################

##########################################################################################
# Dataset
##########################################################################################

	# Load package of the MNIST dataset.
	#
	using MNIST

	# Load the dataset.
	#
	data = traindata()

	# Scale feature values to be between 0 and 1.
	#
	features = data[1]'
	features /= 255.0

	# Size of the dataset.
	#
	N_datapoints = size(features, 1)

##########################################################################################
# Settings
##########################################################################################

	# Number of neurons for the features.
	#
	N_x = 28^2

##########################################################################################
# Methods
##########################################################################################

	# Compute input to the sigmoid function from the probability.
	#
	unsigmoid(y) = log(y./(1-y))

##########################################################################################
# Fit
##########################################################################################

	# Compute mean of the features.
	#
	px = mean(features', 2)

	# Fit bias terms.
	#
	b_x = unsigmoid(px)

	# Chop extreme values.
	#
	b_x[find(b_x .< -25.0)] = -25.0
	b_x[find(b_x .> 25.0)] = 25.0

##########################################################################################
# Save
##########################################################################################

	# Create folder to hold parameters.
	#
	mkpath("bin")

	# Save the parameters.
	#
	writecsv("bin/fit_b_x.csv", b_x)



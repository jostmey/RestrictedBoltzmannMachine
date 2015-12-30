##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-12-23
# Environment: Julia v0.4
# Purpose: Sample restricted Boltzmann machine as a generative model.
##########################################################################################

##########################################################################################
# Packages
##########################################################################################

	# Load tools for sampling N-way outcomes according to a set of weights.
	#
	using StatsBase

##########################################################################################
# Settings
##########################################################################################

	# Schedule for updating the neural network.
	#
	N_samples = 100
	N_passes = 100

	# Number of neurons in each layer.
	#
	N_x = 28^2
	N_z = 10
	N_h = 500

	# Load neural network parameters.
	#
	b_x = readcsv("bin/train_b_x.csv")
	W_xh = readcsv("bin/train_W_xh.csv")
	b_z = readcsv("bin/train_b_z.csv")
	W_zh = readcsv("bin/train_W_zh.csv")
	b_h = readcsv("bin/train_b_h.csv")

	# Randomly initialize each layer.
	#
	px_s = rand(N_x, N_minibatch)
	pz_s = rand(N_z, N_minibatch)
	ph_s = rand(N_h, N_minibatch)

##########################################################################################
# Methods
##########################################################################################

	# Activation functions.
	#
	sigmoid(x) = 1.0./(1.0+exp(-x))
	softmax(x) = exp(x)./sum(exp(x))

	# Sampling methods.
	#
	state(p) = 1.0*(rand(size(p)) .<= p)
	choose(p) = ( y = zeros(size(p)) ; for i=1:size(p, 2) j=sample(WeightVec(p[:,i])) ; y[j,i] = 1.0 end  ; y )

##########################################################################################
# Generate
##########################################################################################

	# Repeatedly sample the model.
	#
	for i = 1:N_samples

		# Sample initial state of each layer.
		#
		x = state(px_s[:,j])
		z = choose(pz_s[:,j])
		h = state(ph_s[:,j])

		# Repeated passes of Gibbs sampling.
		#
		for k = 1:N_passes

			px_s[:,j] = sigmoid(W_xh*h+b_x)
			x = state(px_s[:,j])

			pz_s[:,j] = softmax(W_zh*h+b_z)
			z = choose(pz_s[:,j])

			ph_s[:,j] = sigmoid(W_xh'*x+W_zh'*z+b_h)
			h = state(ph_s[:,j])

		end

	end

##########################################################################################
# Figure
##########################################################################################




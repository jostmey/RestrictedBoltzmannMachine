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

	# Randomly initialize each layer.
	#
	x_s = rand(0.0:1.0, N_x, N_minibatch)
	h_s = rand(0.0:1.0, N_h, N_minibatch)
	z_s = choose(rand(N_z, N_minibatch))

	# Repeatedly sample the model.
	#
	for i = 1:N_samples

		# Repeated passes of Gibbs sampling.
		#
		for k = 1:N_passes

			ph = sigmoid(W_xh'*x_s[:,j]+W_zh'*z_s[:,j]+b_h)
			h_s[:,j] = state(ph)

			px = sigmoid(W_xh*h_s[:,j]+b_x)
			x_s[:,j] = state(px)

			pz = softmax(W_zh*h_s[:,j]+b_z)
			z_s[:,j] = choose(pz)

		end

	end

##########################################################################################
# Figure
##########################################################################################




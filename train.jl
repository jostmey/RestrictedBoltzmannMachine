##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-12-23
# Environment: Julia v0.4
# Purpose: Train restricted Boltzmann machine.
##########################################################################################

##########################################################################################
# Packages
##########################################################################################

	# Load tools for sampling N-way outcomes according to a set of weights.
	#
	using StatsBase

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

	# Copy over the labels.
	#
	labels = data[2]

	# Size of the dataset.
	#
	N_datapoints = size(features, 1)

##########################################################################################
# Settings
##########################################################################################

	# Schedule for updating the neural network.
	#
	N_minibatch = 100
	N_updates = round(Int, N_datapoints/N_minibatch)*500
	N_passes = 3

	# Number of neurons in each layer.
	#
	N_x = 28^2
	N_z = 10
	N_h = 500

	# Parameters of the Guassian prior.
	#
	sigma = 0.1

	# Initialize neural network parameters.
	#
	b_x = sigma*randn(N_x)
	W_xh = sigma*randn(N_x, N_h)
	b_z = sigma*randn(N_z)
	W_zh = sigma*randn(N_z, N_h)
	b_h = sigma*randn(N_h)

	# Initial learning rate.
	#
	alpha = 0.00001

	# Momentum factor.
	#
	momentum = 0.75

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

	# Logarithmic derivative of the prior distribution.
	#
	dlogn(theta) = -theta/sigma^2

##########################################################################################
# Train
##########################################################################################

	# Initialize persistent states.
	#
	x_p = rand(0.0:1.0, N_x, N_minibatch)
	h_p = rand(0.0:1.0, N_h, N_minibatch)
	z_p = choose(rand(N_z, N_minibatch))

	# Holds change in parameters from a minibatch.
	#
	db_x = zeros(N_x)
	dW_xh = zeros(N_x, N_h)
	db_z = zeros(N_z)
	dW_zh = zeros(N_z, N_h)
	db_h = zeros(N_h)

	# Repeatedly update parameters.
	#
	for i = 1:N_updates

		# Collect data-driven samples.
		#
		for j = 1:N_minibatch

			# Randomly select row from the dataset (part of stochastic gradient descent).
			#
			k = rand(1:N_datapoints)

			# Load the features into the visible layer.
			#
			px = features[k,:]'
			x = state(px)

			# Load the labels into the visible layer.
			#
			z = zeros(10)
			z[round(Int, labels[k])+1] = 1.0

			# Gibbs sampling driven by the data.
			#
			ph = sigmoid(W_xh'*x+W_zh'*z+b_h)
			h = state(ph)

			# Summate logarithmic derivative calculated at each sample.
			#
			db_x += x
			dW_xh += x*h'
			db_z += z
			dW_zh += z*h'
			db_h += h

		end

		# Collect samples from the model.
		#
		for j = 1:N_minibatch

			# Continue Gibbs sampling of the model using persistent states.
			#
			for l = 1:N_passes

				px = sigmoid(W_xh*h_p[:,j]+b_x)
				x_p[:,j] = state(px)

				pz = softmax(W_zh*h_p[:,j]+b_z)
				z_p[:,j] = choose(pz)

				ph = sigmoid(W_xh'*x_p[:,j]+W_zh'*z_p[:,j]+b_h)
				h_p[:,j] = state(ph)

			end

			# Summate logarithmic derivative calculated at each sample.
			#
			db_x -= x_p[:,j]
			dW_xh -= x_p[:,j]*h_p[:,j]'
			db_z -= z_p[:,j]
			dW_zh -= z_p[:,j]*h_p[:,j]'
			db_h -= h_p[:,j]

		end

		# Update parameters using stochastic gradient descent.
		#
		b_x += alpha*(dlogn(b_x)+(N_datapoints/N_minibatch)*db_x)
		W_xh += alpha*(dlogn(W_xh)+(N_datapoints/N_minibatch)*dW_xh)
		b_z += alpha*(dlogn(b_z)+(N_datapoints/N_minibatch)*db_z)
		W_zh += alpha*(dlogn(W_zh)+(N_datapoints/N_minibatch)*dW_zh)
		b_h += alpha*(dlogn(b_h)+(N_datapoints/N_minibatch)*db_h)

		# Reset the parameter changes from the minibatch (scale by momentum factor).
		#
		db_x *= momentum
		dW_xh *= momentum
		db_z *= momentum
		dW_zh *= momentum
		db_h *= momentum

		# Decrease the learning rate (part of stochastic gradient descent).
		#
		alpha *= (N_updates-i)/(N_updates-i+1)

		# Periodic checks.
		#
		if i%100 == 0

			# Print progress report.
			#
			println("REPORT")
			println("  Batch = $(round(Int, i))")
			println("  alpha = $(round(alpha, 8))")
			println("")
			flush(STDOUT)

		end

	end

##########################################################################################
# Save
##########################################################################################

	# Create folder to hold parameters.
	#
	mkpath("bin")

	# Save the parameters.
	#
	writecsv("bin/train_b_x.csv", b_x)
	writecsv("bin/train_W_xh.csv", W_xh)
	writecsv("bin/train_b_z.csv", b_z)
	writecsv("bin/train_W_zh.csv", W_zh)
	writecsv("bin/train_b_h.csv", b_h)


##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-12-23
# Environment: Julia v0.4
# Purpose: Train restricted Boltzmann machine.
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

	# Number of neurons in each layer.
	#
	Nv = 28^2+10
	Nh = 500

	# Standard deviation of Gaussian prior over the parameters.
	#
	sigma = 0.1

	# Initialize neural network parameters.
	#
	b = sigma*randn(Nv)
	W = sigma*randn(Nv, Nh)
	a = sigma*randn(Nh)

	# Persistent states.
	#
	persistent = rand(0.0:1.0, Nv, N_minibatch)

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

	# Sampling methods.
	#
	state(p) = 1.0*(rand(size(p)) .<= p)

##########################################################################################
# Train
##########################################################################################

	# Holds change in parameters from a minibatch.
	#
	db = zeros(Nv)
	dW = zeros(Nv, Nh)
	da = zeros(Nh)

	# Repeatedly update parameters.
	#
	for i = 1:N_updates

		# Collect data-driven samples.
		#
		for j = 1:N_minibatch

			# Randomly load item from the dataset (part of stochastic gradient descent).
			#
			k = rand(1:N_datapoints)

			x = features[k,:]'

			z = zeros(10)
			z[round(Int, labels[k])+1] = 1.0

			# Gibbs sampling.
			#
			pv = [x;z]
			v = state(pv)

			ph = sigmoid(W'*v+a)
			h = state(ph)

			# Summate derivative calculated at each sample.
			#
			db += v
			dW += v*h'
			da += h

		end

		# Collect samples from the model.
		#
		for j = 1:N_minibatch

			# Load persistent state.
			#
			v = persistent[:,j]

			# Gibbs sampling.
			#
			ph = sigmoid(W'*v+a)
			h = state(ph)

			pv = sigmoid(W*h+b)
			v = state(pv)

			# Save updated persistent state.
			#
			persistent[:,j] = v

			# Add derivative calculated at this sample.
			# Summate derivative calculated at each sample.
			db -= v
			dW -= v*h'
			da -= h

		end

		# Update parameters using stochastic gradient descent.
		#
		b += alpha*((N_datapoints/N_minibatch)*db-b/sigma^2)
		W += alpha*((N_datapoints/N_minibatch)*dW-W/sigma^2)
		a += alpha*((N_datapoints/N_minibatch)*da-a/sigma^2)

		# Reset the parameter changes from the minibatch (scale by momentum factor).
		#
		db *= momentum
		dW *= momentum
		da *= momentum

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
	writecsv("bin/train_b.csv", b)
	writecsv("bin/train_W.csv", W)
	writecsv("bin/train_a.csv", a)

	# Save persistent state.
	#
	writecsv("bin/train_persistent.csv", persistent)


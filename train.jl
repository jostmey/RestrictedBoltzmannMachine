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

	# Number of neurons in each layer.
	#
	N_x = 28^2
	N_z = 10
	N_h = 500

	# Standard deviation of Gaussian prior over the parameters.
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

##########################################################################################
# Train
##########################################################################################

	# Persistent states.
	#
	x_persist = rand(0.0:1.0, N_x, N_minibatch)
	z_persist = choose(rand(N_z, N_minibatch))

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

			# Summate derivative calculated at each sample.
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

			# Load persistent state.
			#
			x = x_persist[:,j]
			z = z_persist[:,j]

			# Gibbs sampling driven by the model.
			#
			ph = sigmoid(W_xh'*x+W_zh'*z+b_h)
			h = state(ph)

			px = sigmoid(W_xh*h+b_x)
			x = state(px)

			pz = softmax(W_zh*h+b_z)
			z = choose(pz)

			# Save persistent state.
			#
			x_persist[:,j] = x
			z_persist[:,j] = z

			# Summate derivative calculated at each sample.
			#
			db_x -= x
			dW_xh -= x*h'
			db_z -= z
			dW_zh -= z*h'
			db_h -= h

		end

		# Update parameters using stochastic gradient descent.
		#
		b_x += alpha*((N_datapoints/N_minibatch)*db_x-b_x/sigma^2)
		W_xh += alpha*((N_datapoints/N_minibatch)*dW_xh-W_xh/sigma^2)
		b_z += alpha*((N_datapoints/N_minibatch)*db_z-b_z/sigma^2)
		W_zh += alpha*((N_datapoints/N_minibatch)*dW_zh-W_zh/sigma^2)
		b_h += alpha*((N_datapoints/N_minibatch)*db_h-b_h/sigma^2)

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
			println("PARAMETERS")
			println("  Mean(b_x) = $(round(mean(b_x), 5)), Max(b_x) = $(round(maximum(b_x), 5)), Min(b_x) = $(round(minimum(b_x), 5))")
			println("  Mean(W_xh) = $(round(mean(W_xh), 5)), Max(W_xh) = $(round(maximum(W_xh), 5)), Min(W_xh) = $(round(minimum(W_xh), 5))")
			println("  Mean(b_z) = $(round(mean(b_z), 5)), Max(b_z) = $(round(maximum(b_z), 5)), Min(b_z) = $(round(minimum(b_z), 5))")
			println("  Mean(W_zh) = $(round(mean(W_zh), 5)), Max(W_zh) = $(round(maximum(W_zh), 5)), Min(W_zh) = $(round(minimum(W_zh), 5))")
			println("  Mean(b_h) = $(round(mean(b_h), 5)), Max(b_h) = $(round(maximum(b_h), 5)), Min(b_h) = $(round(minimum(b_h), 5))")
			println("UPDATES")
			println("  Mean(db_x) = $(round(mean(db_x), 5)), Max(db_x) = $(round(maximum(db_x), 5)), Min(db_x) = $(round(minimum(db_x), 5))")
			println("  Mean(dW_xh) = $(round(mean(dW_xh), 5)), Max(dW_xh) = $(round(maximum(dW_xh), 5)), Min(dW_xh) = $(round(minimum(dW_xh), 5))")
			println("  Mean(db_z) = $(round(mean(db_z), 5)), Max(db_z) = $(round(maximum(db_z), 5)), Min(db_z) = $(round(minimum(db_z), 5))")
			println("  Mean(dW_zh) = $(round(mean(dW_zh), 5)), Max(dW_zh) = $(round(maximum(dW_zh), 5)), Min(dW_zh) = $(round(minimum(dW_zh), 5))")
			println("  Mean(db_h) = $(round(mean(db_h), 5)), Max(db_h) = $(round(maximum(db_h), 5)), Min(db_h) = $(round(minimum(db_h), 5))")
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


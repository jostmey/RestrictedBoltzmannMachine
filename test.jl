##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-12-23
# Environment: Julia v0.4
# Purpose: Test restricted Boltzmann machine as a classifier.
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
	data = testdata()

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
	N_samples = 100
	N_passes = 10

	# Number of neurons in each layer.
	#
	Nv = 28^2+10
	Nh = 500

	# Initialize neural network parameters.
	#
	b = readcsv("bin/train_b.csv")
	W = readcsv("bin/train_W.csv")
	a = readcsv("bin/train_a.csv")

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
	choose(p) = ( y = zeros(size(p)) ; i = sample(WeightVec(p[:])) ; y[i] = 1.0 ; y )

##########################################################################################
# Test
##########################################################################################

	# Print header.
	#
	println("RESPONSES")

	# Track percentage of guesses that are correct.
	#
	N_correct = 0.0
	N_tries = 0.0

	#
	#
	for i = 1:N_datapoints

		#
		#
		Ev = zeros(10)

		#
		#
		for j = 1:N_samples

			# Randomly load item from the dataset (part of stochastic gradient descent).
			#
			x = features[i,:]'

			z = zeros(10)
			z[rand(1:10)] = 1.0

			# Gibbs sampling.
			#
			pv = [x;z]
			v = state(pv)

			for k = 1:N_passes

				ph = sigmoid(W'*v+a)
				h = state(ph)

				pv[28^+1:end] = softmax(W[28^2+1:end,:]*h+b[28^2+1:end])
				v[28^+1:end] = choose(pv[28^+1:end])

			end

			#
			#
			Ev += pv[28^+1:end]/N_samples

		end

		# Update percentage of guesses that are correct.
		#
		guess = findmax(Ev)[2]-1
		answer = round(Int, labels[i])
		if guess == answer
			N_correct += 1.0
		end
		N_tries += 1.0

		# Print response.
		#
		println("  i = $(i), Guess = $(guess), Answer = $(answer)")

	end

##########################################################################################
# Results
##########################################################################################

	# Print progress report.
	#
	println("SCORE")
	println("  Correct = $(round(100.0*N_correct/N_tries, 5))%")
	println("")


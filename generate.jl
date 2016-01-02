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

	# Load tools for saving images.
	#
	using Images

##########################################################################################
# Settings
##########################################################################################

	# Schedule for updating the neural network.
	#
	N_equilibrate = 1000
	N_samples = 100

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
	px_s = rand(N_x, N_samples)
	pz_s = rand(N_z, N_samples)
	ph_s = rand(N_h, N_samples)

	# Image dimensions.
	#
	height = 28
	width = 28

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
	choose(p) = ( y = zeros(size(p)) ; for i = 1:size(p, 2) j = sample(WeightVec(p[:,i])) ; y[j,i] = 1.0 end  ; y )

##########################################################################################
# Generate
##########################################################################################

	# Repeatedly sample the model.
	#
	for i = 1:N_samples

		# Sample initial state of each layer.
		#
		x = state(px_s[:,i])
		z = choose(pz_s[:,i])
		h = state(ph_s[:,i])

		# Repeated passes of Gibbs sampling.
		#
		for j = 1:N_equilibrate

			px_s[:,i] = sigmoid(W_xh*h+b_x)
			x = state(px_s[:,i])

			pz_s[:,i] = softmax(W_zh*h+b_z)
			z = choose(pz_s[:,i])

			ph_s[:,i] = sigmoid(W_xh'*x+W_zh'*z+b_h)
			h = state(ph_s[:,i])

		end

	end

##########################################################################################
# Figure
##########################################################################################

	# Samples will rendered in tiles by rows and columns.
	#
	N_columns = ceil(Int, sqrt(N_samples))
	N_rows = ceil(Int, N_samples/N_columns)

	# Pixels on which the tiles will be rendered.
	#
	pixels = zeros(height*N_rows, width*N_columns)

	# Index of the tiles.
	#
	k = 1

	# Loop over the rows of the tiles.
	#
	for i = 1:N_rows

		# Loop over the columns of the tiles.
		#
		for j = 1:N_columns

			# Indices for selecting the current tile.
			#
			is = (i-1)*height+1:i*height
			js = (j-1)*width+1:j*width

			# Copy the sample into the tile.
			#
			pixels[is, js] = reshape(px_s[:,k], height, width)

			# Update the index for the tile.
			#
			k += 1
			if k > N_samples
				break
			end

		end

	end

	# Create grayscale image with black background.
	#
	figure = convert(Image, pixels)

	# Save the tiles as an image file.
	#
	save("generate.png", figure)


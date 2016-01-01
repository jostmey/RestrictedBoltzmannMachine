# --- WORK IN PROGRESS ---

## Description

Example scripts for a restricted Boltzmann machine (RBM), which is a type of generative model, have been written from scratch. No machine learning packages are used, providing an example of how to implement the underlying algorithms of an artificial neural network. The code is written in the Julia, a programming language with a syntax similar to Matlab.

The generative model is trained on both the features and labels of the MNIST dataset of handwritten digits. Samples drawn from the model are nearly indistinguishable from handwritten digits. The model is then used as a classifier by loading the features and running a Markov chain until the model has reached equilibrium, at which point the expected value of the label is tabulated. The model correctly classifies XX % of the handwritten digits in the test dataset.

## Download

* Download: [zip](https://github.com/jostmey/RestrictedBoltzmannMachine/zipball/master)
* Git: `git clone https://github.com/jostmey/RestrictedBoltzmannMachine`

## Requirements

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The scripts have been developed using version 0.4 and do not work on previous versions of Julia.

The scripts require several modules, which have to be installed in the Julia environment. To add them, launch `julia` and run the following commands.

`Pkg.add("MNIST")`  
`Pkg.add("StatsBase")`  
`Pkg.add("Images")`

The first package contains the MNIST dataset of handwritten digits. The seconds package contains a tool for sampling from a set of weighted choices. The last package is used to render images. The first time the image package is run it may ask to install additional software.

## Run

Fitting the bias terms of the neurons representing the features using the dataset before training the neural network greatly improves the results. After fitting, the neural network can be trained, which can last anywhere from a few days to several weeks. To start the process, run the following commands.

`julia fit.jl > fit.out`  
`julia train.jl > train.out`

The scripts will automatically create a folder called `bin` where the neural network parameters will be saved. At this point, the neural network will be ready to use. To generate samples from the model, run the following command.

`julia generate.jl > generate.out`

The script may need to be run twice if the image package needs to install additional software. A sequence of samples will be saved in the image file `generate.png`. To classify the handwritten digits in the test set, run the following command.

`julia classify.jl > classify.out`

The percentage of correct answers will be written at the end of the text file `test.out`.

## Performance

This package is not written for speed. It is meant to serve as a working example of an artificial neural network. As such, there is no GPU acceleration. Training using only the CPU can take days or even weeks. The training time can be shortened by reducing the number of updates, but this could lead to poorer performance on the test data. Consider using an exising machine learning package when searching for a deployable solution.

## Theory

###### Model

A restricted Boltzmann machines (RBM)s is a special type of Boltzmann machine, which is a generative model that can be trained to represent a dataset as a joint probability distribution. What this means is that if you draw a statistical sample from the model, it should look a new item was just added to the dataset. In otherwords, the model learns how to "make up" new examples based off of the dataset.

In a Boltzmann machine, the neurons are divided into two layers: A visible layer that represents the sensory input and a hidden layer that is determined solely by its connections to the visible layer. A Boltzmann machine can be though of as a probability distribution described by an energy function `E(v,h)`, where `v` and `h` are vectors used to represent the state of the neurons in the visible and hidden layers, respectively. The energy function takes the state of each neuron in the neural network and returns a scalar value. The probability of observing the neurons in a specific state is proportional to `exp(-E(v,h))` (assuming the temperature factor is one). The negative sign in front of the energy means that if the energy is high the probability will be low. Because the probability is proportional to `exp(-E(v,h))`, calculating an exact probability requires a normalization constast. The normalization constant is denoted `Z` and is called the partition function. `Z` is the sum of `exp(-E(v,h))` over all possible states of `v` and `h`, so in general calculating `Z` exactly is intractable.

The energy function `E(v,h) = -v'*W*h - b'*v - a'*h` is used in this model. The matrix `W` describes the weights of the connections between neurons in the visible and hidden layers, and the vectors `b` and `a` describe the biases of the neurons in the visible and hidden layers, respectively. The operator `'` transposes the vector -- in this way we always end up with a scalar value after each multiplication.

###### Sampling

Gibbs sampling is used to update the value of each neuron. Assuming that each neuron has only two states `0` and `1`, we can calculate the probability of the i<sup>th</sup> neuron being on given the state of the rest of the neural network. The probability can be found using the energy function `E(v,h)` by flipping the i<sup>th</sup> neuron so that it is on, which we will write as `E`<sub>`i=1`</sub>`(v,h)`. The probability that the <sup>i</sup> neuron is on is therefore proportional to `exp(-E`<sub>`i=1`</sub>`(v,h))`, and the total probability that the neuron is either on or off is proportional to `exp(-E`<sub>`i=0`</sub>`(v,h)) + exp(-E`<sub>`i=1`</sub>`(v,h))`. Because the total probability is always `1`, we can divide the probability by the total probability and get the same answer. What this means is that normalization constant `Z` will cancel out and the probability that the i<sup>th</sup> neuron is on will be `exp(-E`<sub>`i=1`</sub>`(v,h))` divided by `exp(-E`<sub>`i=0`</sub>`(v,h)) + exp(-E`<sub>`i=1`</sub>`(v,h))`. The term can be written as `Sigmoid(-E`<sub>`i=0`</sub>`(v,h) - E`<sub>`i=1`</sub>`(v,h))` by dividing the top and bottom by the numerator, where `Sigmoid` is the logistic function. The neural network is run by taking each neuron and calculating its probability of being having a value of `1`. The neuron is then randomly assigned a new state of either `0` or `1` based on this probability. Each neuron must be updated separately. After updating each neuron in the neural network a sufficient number of times, the model will reach what is called equilibrium.

In a RBM, connections between neurons in the same layer are removed. The only connections that exist are between the visible and hidden layers. Without cross connections, neurons in the same layer depend only on the state of the neurons in the other layer. Therefore, neurons in the same layer can be updated simultaneously in a single sweep. Given the state of the neurons in the visible layer, the state of the neurons in the hidden layer can be calculated in a single step, and vice versa.

###### Training

Boltzmann machines can be fit to a dataset by obtaining a maximum-likelihood estimation (MLE) of the neural network parameters. Typically, gradient ascent is used to optimize the log-likelihood of the model (taking the log of the likelihood does not effect the optimum points). The derivative is used to make small adjustments to parameters of the neural networks up the gradient. Changes are made to the weights in small, discrete steps determined by the value of the *learning rate*.

The derivative of the log-likelihood function for a neural nework is remarkably simple. It is the difference between two expectations. The first expectation is `<-dE(v,h)/du)>`<sub>`DATA`</sub>, where `u` is the parameter, is calculated over the dataset. The result is subtracted by the second expectation `<-dE(v,h)/du>`<sub>`MODEL`</sub>. Each of the two expectations can be estimated by collecting a set of samples, a process that is called Markov Chain Monte-Carlo. For the first expectation, the neurons in the visible layer is set to the values of an item in the dataset and the neurons in the hidden layer are updated until equilibrium is reached. For a RBM, only a single sweep is required to reach equilibrium. The latter expectation is harder to sample. Sampling from the model requires starting from a random configuration of the neurons in the visible and hidden layers, and sampling until the model reaches equilibrium. Because none of the neurons are clamped, it can take awhile to reach equilibrium. One way to efficiently sample the model is to maintain a set of persistent states. Each persistent states is initialized by first running the neural network until it reaches equilibrium. Then, after each time the parameters are adjusted during the gradient optimization procedure, the persistent states are updated. Because the parameters only changed a small amount, the persistent states will be close to equilibrium, which means that each neuron will only need to be updated a few times to reach equilibrium again.

Ideally, both expectations should be sampled over all possible states, a process that is too expensive in practice. Instead, stochastic gradient descent is used. With stochastic gradient descent, a subset of examples randomly drawn are used to update the weights. To compensate for using only a subset of the total number of training examples, the values from the subset are scaled by the ratio of total training examples over the number of actual training examples used in the subset. The procedure will preserve the true objective function provided that the learning rate is decreased at each iteration following a specific schedule. The schedule used here is an approximation--the learning rate is decreased following a linear progression.

A problem with gradient optimization methods  is that the fitting procedure may not find the global maxima of the log-likelihood. A momentum term is included to help escape from local minimums.

###### Prior

The neural network contains nearly `400 000` parameters. To prevent overfitting, a prior is added over the parameters to reduce the hypothesis space, which in this case is a Gaussian distribution centered at zero. This changes the fitting procedure of the parameters, so that we are now obtaining a maximum a posteriori probability (MAP) estimate instead of a MLE. Instead of following the gradient of the log-likelihood function, we must now also add the log-derivative of the prior distribution.

Normally the training data should be split into a training and validation set. Multiple versions of the model are then trained on training set each using different learning rates, momentum factors, prior distributions, and number of updates. The model that scores the highest on the validation set is then used on the test data. The use of a validation set means that the test data is never seen while selecting the best model, which would be cheating. That said, no validation set is used in this example because the model was never refined--only one version of the model was trained. This model was then tested directly on the test data.

###### Model

In this example, the features and labels of the MNIST dataset of handwritten digits was loaded into the visible layer. This way, the neural network can be used as a generative model to create samples that resemble what it is trained on while at the same time allowing it to be used as a classifier. To represent the features of the MNIST dataset, the intensity of each pixel is used as the mean value of a Bernoulli distirubion. Samples collected from the Bernoulli distribution, which are binary, can then be loaded into the visible layer. To represent the layer, a softmax unit is used, which is a multinomial generalization of a neuron. The use of a softmax unit does not change the gradient optimization procedure used to fit the parameters.

To run as a generative model, each of the neurons must be updated until the neural network reaches equilibrium. At this point, the neurons in the visible layer that corresponded to the features are read and their values used to generate an image. The resulting images resemble the types of things found in the training data.

To run as a classifier, we must compute the expected value of the label given the features. To generate samples for this expectation, the features and a random label are loaded into the visible layer. The neurons in the hidden layer along with the neuron represneting the label are repeatedly updated until equilibrium is reached. The value of the label after reaching equilibrium is then used to help calculate the expectation.

###### References

[comment]: # (BIBLIOGRAPHY STYLE: MLA)

1. Ackley, David H., Geoffrey E. Hinton, and Terrence J. Sejnowski. "A learning algorithm for boltzmann machines*." Cognitive science 9.1 (1985): 147-169.
2. Smolensky, Paul. "Chapter 6: Information Processing in Dynamical Systems: Foundations of Harmony Theory. Processing of the Parallel Distributed: Explorations in the Microstructure of Cognition, Volume 1: Foundations." (1986).
3. Robbins, Herbert, and Sutton Monro. "A stochastic approximation method." The annals of mathematical statistics (1951): 400-407.
4. Tieleman, Tijmen. "Training restricted Boltzmann machines using approximations to the likelihood gradient." Proceedings of the 25th international conference on Machine learning. ACM, 2008.

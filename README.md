## Description

Example scripts for a restricted Boltzmann machine (RBM), which is a type of generative model, have been written from scratch. No machine learning packages are used, providing an example of how to implement the underlying algorithms of an artificial neural network. The code is written in the Julia, a programming language with a syntax similar to Matlab.

The generative model is trained on both the features and labels of the MNIST dataset of handwritten digits. Samples drawn from the model are nearly indistinguishable from handwritten digits. The model is then used as a classifier by loading the features and running a Markov chain until the model has reached equilibrium, at which point the expected value of the label is tabulated. The model correctly classifies XX % of the handwritten digits in the test dataset.

## Download

* Download: [zip](https://github.com/jostmey/RestrictedBoltzmannMachine/zipball/master)
* Git: `git clone https://github.com/jostmey/RestrictedBoltzmannMachine`

## Requirements

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The scripts have been developed using version 0.4 and do not work on previous versions of Julia.

The scripts require several modules, which have to be installed in the Julia environment. To add them, launch `julia` and run `Pkg.add("MNIST")` and `Pkg.add("StatsBase")`.

## Run

Training the neural network can take several days or even weeks. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To generate samples from the model, run the following command.

`julia generate.jl > generate.out`

A sequence of samples will be saved in the image file `generate.???`. To classify the handwritten digits in the test set, run the following command.

`julia classify.jl > classify.out`

The percentage of correct answers will be written at the end of the text file `test.out`.

## Performance

This package is not written for speed. It is meant to serve as a working example of an artificial neural network. As such, there is no GPU acceleration. Training using only the CPU can take days or even weeks. The training time can be shortened by reducing the number of updates, but this could lead to poorer performance on the test data. Consider using an exising machine learning package when searching for a deployable solution.

## Theory

###### Model

A restricted Boltzmann machines (RBM)s is a special type of Boltzmann machine, which is a generative model that can be trained to represent a dataset as a joint probability distribution. What this means is that if you draw a statistical sample from the model, it should look a new item was just added to the dataset. In otherwords, the model learns how to "make up" new examples based off of the dataset.

In a Boltzmann machine, the neurons are divided into two layers: A visible layer that represents the sensory input and a hidden layer that is determined solely by its connections to the visible layer. A Boltzmann machine can be though of as a probability distribution described by an energy function `E(v,h)`, where `v` and `h` are vectors used to represent the state of the neurons in the visible and hidden layers, respectively. The energy function takes the state of each neuron in the neural network and returns a scalar value. The probability of observing the neurons in a specific state is proportional to `exp(-E(v,h))` (assuming the temperature factor is `1`). The negative sign in front of the energy means that if the energy is high the probability will be low. Because the probability is proportional to `exp(-E(v,h))`, calculating an exact probability requires a normalization constast. The normalization constant is denoted `Z` and is called the partition function. `Z` is the sum of `exp(-E(v,h))` over all possible states of `v` and `h`, so in general calculating `Z` exactly is intractable.

The form of the energy function used in this model is `E(v,h) = v'*W*h + b'*v + a'*h`. `W` is matrix describing the weights of the connections between neurons in the visible and hidden layers, and `b` and `a` are vectors that describe the biases of the neurons in visible and hidden layers, respectively. The `'` tells us to transpose the vector so that we always end up with a scale after each multiplication.

###### Sampling

Gibbs sampling is used to update the value of each neuron. Assuming that each neuron has only two states `0` and `1`, we can calculate the probability of the i<sup>th</sup> being on given the state of the entire neural network. The probability that the i<sup>th</sup> neuron is on is proportional to `exp(-E(v,h|i=1))` while the total probability that the i<sup>th</sup>neuron is either on or off is proportional to `exp(-E(v,h|i=0)) + exp(-E(v,h|i=1))`. Because the total probability is always `1`, we can divide the probability by the total probability and get the same answer. What this means is that normalization constant `Z` will cancel out, so that the probability that the i<sup>th</sup> neuron is on is `exp(-E(v,h|i=1))` divided by `exp(-E(v,h|i=0)) + exp(-E(v,h|i=1))`. The division simplifies to `Sigmoid( E(v,h|i=1) - E(v,h|i=1) )`. The neural network is run by taking each neuron one at a time and calculating its probability of having a value of `1`. The neuron is then randomly assigned a new state of either `0` or `1` based on this probability. After updating each neuron a sufficient number of time the neural network will eventually reach what is called equilibrium. 

In a RBM, connections between neurons in the same layer are removed. The only connections that exist are between the visible and hidden layers. Because the neurons in the same layer are statistically independent of each other given the state of the other layer, neurons in the same layer can be updated simultaneously in a single sweep. Sampling in a RBM becomes quite simple. Given the state of the neurons in the visible layer, the state of the neurons in the hidden layer can be calculated in a single step, and vice versa.

###### Training



###### Prior



###### Results




WORK IN PROGRESS... CHECK BACK LATER...


## Description

Example scripts for a restricted Boltzmann machine (RBM), which is a type of generative model, have been written from scratch. No machine learning packages are used, providing an example of how to implement the underlying algorithms of an artificial neural network. The code is written in the Julia, a programming language with a syntax similar to Matlab.

The generative model is trained on both the features and labels of the MNIST dataset of handwritten digits. Samples drawn from the model are nearly indistinguishable from handwritten digits. The model is then used as a classifier by loading the features and running a Markov chain until the model has reached equilibrium, at which point the expected value of the label is tabulated. The model correctly classifies XX % of the handwritten digits in the test dataset.

## Download

* Download: [zip](https://github.com/jostmey/RestrictedBoltzmannMachine/zipball/master)
* Git: `git clone https://github.com/jostmey/RestrictedBoltzmannMachine`

## REQUIREMENTS

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The scripts have been developed using version 0.4 and do not work on previous versions of Julia.

The scripts require several modules, which have to be installed in the Julia environment. To add them, launch `julia` and run `Pkg.add("MNIST")` and `Pkg.add("StatsBase")`.

## RUN

Training the neural network can take several days or even weeks. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To generate samples from the model, run the following command.

`julia generate.jl > generate.out`

A sequence of samples will be saved in the image file `generate.???`. To classify the handwritten digits in the test set, run the following command.

`julia classify.jl > classify.out`

The percentage of correct answers will be written at the end of the text file `test.out`.

## THEORY

###### Model

A restricted Boltzmann machines (RBM)s is a special type of Boltzmann machine, which is a generative model that can be trained to represent a dataset as a joint probability distribution. What this means is that if you draw a statistical sample from the model, it should look a new item was just added to the dataset. In otherwords, the model learns how to "make up" new examples based off of the dataset.

In a Boltzmann machine, the neurons are divided into two layers: A visible layer that represents the sensory input and a hidden layer that is determined solely by its connections to the visible layer. A Boltzmann machine can be though of as a probability distribution described by an energy function `E(v,h)`, where `v` and `h` are vectors used to represent the state of the neurons in the visible and hidden layers, respectively. The energy function takes the state of each neuron in the neural network and returns a scalar value. The probability of observing the neurons in a specific state is proportional to `exp(-E(v,h))` (assuming the temperature factor is `1`). The negative sign in front of the energy means that if the energy is high the probability will be low. Because the probability is proportional to `exp(-E(v,h))`, calculating an exact probability requires a normalization constast. The normalization constant is denoted `Z` and is called the partition function. `Z` is the sum of `exp(-E(v,h))` over all possible states of `v` and `h`, so in general calculating `Z` exactly is intractable.




Gibbs sampling is typically used to draw samples from the model. 

Connections are restricted to neurons in different layers. No connections are found between neurons in the same layer. This makes inference simple. Given the state of the neurons in the visible layer, the state of the neurons in the hidden layer can be inferred in a single step, and vice versa.











WORK IN PROGRESS... CHECK BACK LATER...


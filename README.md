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

In a restricted Boltzmann machine (RBM), the neurons are divided into two layers: The visible layer that receive sensory input and the hidden layer, which is connected to the visible layer. No connections exist between neurons in the same layer, which is why this type of Boltzmann machine is said to be "restricted". The model is generative meaning that it will learn a probability distribution over its sensory input. The model is typically described using an energy function `E(v,h)=v`<sup>`T`</sup>`*W*h+b*v+a*h`. Here `v` and `h` are vectors representing the state of the neurons in the visible and hidden layers, respectively. `W` is matrix describing the weights of the connections between neurons in the visible and hidden layers, and `b` and `a` are vectors that describe the biases of the neurons in visible and hidden layers, respectively. After calculating the energy, a scalar value is obtained. The energy function describes a probability distribution. The probability distribution `P(v,h)` is proportional to `exp(-E(v,h))`. To calculate a probability, a normalization constanst is needed, which is called the partition function `Z`. `Z` is the sum over all possible values, and is given by `Z=Sum`<sub>`v,h`</sub>`{ exp(-E(v,h)) }`

Gibbs sampling is typically used to draw samples from the model. 












WORK IN PROGRESS... CHECK BACK LATER...


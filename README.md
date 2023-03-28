# Analysis and usage of molearn

[molearn](https://github.com/Degiacomi-Lab/molearn) is a generative neural network trainable with protein conformations. This repository contains the following Jupyter notebooks. 

* `molearn analysis.ipynb`: a tutorial showing in detail how to interact with a trained neural network, and how to analyse its performance. This notebook shows how analysis is carried out in `molearn.analysis.MolearnAnalysis`.
* `molearn_GUI.ipynb`: a demonstration of a graphical user interface used to display a neural network latent space, and exploit it to generate protein conformations.
* `minimal_example.ipynb`: the minimal lines of code required to load a trained neural network and the data it was trained with, gather analysis data, and display them graphically.

If you use molearn in your work, please cite:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

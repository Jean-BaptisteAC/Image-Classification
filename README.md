# :camera: Image Classification

This project aims to classify images from the MNIST-number dataset using a simple Convolution Neural Network.
One can understand that using only a few number of convolution layers can yeld to very good results. 

We will also use dimension reduction and visualisation techniques like **Principal Component Analysis** (PCA) and **t-distributed Stochastic Neighbor Embedding** (t-SNE) in order to visualise the process of classification within the layers of our neural network.

## Using the code

You can use the jupyter version of the project with the file ```Image Classification.ipynb```, or the python file ```Image Classification.py```.

Other useful documents can be found in the **Ressources** Folder.


## Model 

The following figure gives us a scheme of the neural network, comprising Convolution Layers, ReLU layers, Max Pooling, Batch Normalization and Dense Layers for dimension reduction. 

![Image](Ressources/Diagramm.drawio.png)

*The written sizes are corresponding to the output format of data after each layer*

This architecture takes as input an 28x28 pixels image and returns a vector of size 10 corresponding to the probabilities of classes, through a SoftMax activation function. The Flatten Layers exist in order to easily compute the PCA and t-SNE operations.


## Results

## PCA and t-SNE visualisation


CNN Architecture:
Input image -> Convolutions -> Pooling -> Fully Connected

Convolution - helps the computer digest what the image is actually showing it
Uses filters to render groups of pixels of the image to get the general sum of information in the relative area by extracting the important features of respective areas of the image.
As a function, it takes in a filter size and a stride(the number of pixels skipped before taking in the next group of pixels) and outpits a feature map or activation map. A feature/activation map is essentially the result of what was extracted from the filtering process of many groups of pixels (filter_size by filter_size)
Larger datasets will require larger convolution networks.

Pooling - after the image is registered, pooling pools the average "value" of the pixels together and either selects the maximum value (Max Pooling) or the average value (Average Pooling)
Max Pooling: Preserves detected features, most commonly used
Average Pooling: Downsamples feature map, Used in LeNet(one of the earliest CNN ever made)

Though seemingly similar, pooling and convolutional layers are used in conjunction. Pooling reduces the spatial dimensions of feature maps while convolutional lauers extract significant features of the input.

Fully Connected (FC) - typically the last layer of the CNN architecture used to present final objectives and scores
Each input is connected to all neurons to create a flattened input (converting multi dimenional outputs(2d/3d vectors like images) from the previous layer to a 1D array).

When tracking the progress of your neural network, your loss should always be decreasing, otherwise, you need a deeper neural network.
Make the commented changes to develop a deeper cnn and observe higher validation accuracies, improving model performance.
- Added 2 more convolution layers
- Added 1 more dense layer and increased the number of units (neurons)
- Increased total params from 1.6m to ~ 5.7m (cnn.summary)

Other terms:
Batch Size - The number of samples you feed into your model each iteration of the training process. It determines how often the model parameters are updated based on the gradient of the loss function. A larger batch size means for data per update, but also more memory and computation requirements. A smaller batch size means less data per update, but also more noise and variance in the gradient.
Overfitting - Occurs when an algorithm fits too closely to the training data, resulting in a model that is crippled beyond the scale of the training data. Keras prevents this by randomly dropping connections between the hidden layers to prevent the network from learning the training data too well.
Epochs - How many times a dataset passes through an algorithm. It refers to one entire passing of the training data through the algorithm.
Loss - A quantified value that represents the difference between the predicted output and the actual output of a machine learning algorithm.

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

Dataset from kaggle food and vegetable set
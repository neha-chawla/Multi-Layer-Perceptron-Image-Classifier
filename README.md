# Multi-Layer Perceptron Image Classifier

Objective: I've implemented a multi-layer perceptron (MLP) neural network without using any machine learning libraries (tensorflow (v1&v2), caffe, pytorch, torch, cxxnet, mxnet, etc.) and used it to classify hand-written digits as shown in the image below. I've implemented feedforward/backpropagation as well as the training process using the data below.

<img width="509" alt="Screenshot 2024-11-25 at 5 34 21 PM" src="https://github.com/user-attachments/assets/be2eab03-fd0d-42c5-8933-bee2d181b09b">

Data (MNIST dataset: ​http://yann.lecun.com/exdb/mnist/​):
- Training set images​, which contains 60,000 28 × 28 grayscale training images, each representing a single handwritten digit.
- Training set labels​, which contains the associated 60,000 labels for the training images.
- Test set images​, which contains 10,000 28 × 28 grayscale testing images, each representing a single handwritten digit.
- Test set labels​, which contains the associated 10,000 labels for the testing images.

File 1 and 2 are the training set. File 3 and 4 are the test set. Each training and test instance in the MNIST database consists of a 28 × 28 grayscale image of a handwritten digit and an associated integer label indicating the digit that this image represents (0-9). Each of the 28 × 28 = 784 pixels of each of these images is represented by a single 8-bit color channel. Thus, the values each pixel can take on range from 0 (completely black) to 255 (​28 − 1, ​completely white). If you are interested, the raw MNIST format is described in ​http://yann.lecun.com/exdb/mnist/​.

Result: 89% accuracy

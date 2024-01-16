# Plain Artificial Intelligence
PAI - Plain Neural Network, written in C#, from scratch.
This is a simple neural network with sample code to train on MNIST, as a learning experience, it is easily configurable in code and can be reworked to fit any need.

By default, it utilizes Leaky ReLU combined with Softmax as the activation functions, and Binary Cross Entropy as the loss function. However, other activation functions and loss functions are also included.
After 36 epochs, it reaches 100% accuracy on the MNIST training data and 98.01% on the test data.

The network can be saved and loaded using Binary Serialization.
In the future, there are plans to make this project multithreaded.
Feel free to make any changes to the code that may enhance the network's learning capabilities :)


The MNIST Dataset used for this project can be downloaded here: [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

<img width="510" alt="paipic" src="https://github.com/AreOlsen/Plain-Neural-Network/assets/58704301/eaff9bd8-4ed0-4eb0-a867-408890094887">


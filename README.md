CNN Image Classification for MNIST Dataset
Overview
This project demonstrates the implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 testing images of digits (0-9), each sized 28x28 pixels.

Features
Preprocessing of the MNIST dataset
Construction of a CNN model
Training and evaluation of the model
Visualization of training results and predictions
Requirements
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
Installation
Clone this repository:

sh
Copy code
git clone https://github.com/Wizbik27/CNN-CONVOLUTIONAL-NEURAL-NETWORK
cd mnist-cnn
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Run the script to preprocess the data, build, train, and evaluate the CNN model:

sh
Copy code
python mnist_cnn.py
To visualize the training process and the results, use the provided Jupyter notebook:

sh
Copy code
jupyter notebook mnist_cnn.ipynb
Project Structure
mnist_cnn.py: Main script to run the entire process.
mnist_cnn.ipynb: Jupyter notebook for step-by-step execution and visualization.
requirements.txt: List of required packages.
Model Architecture
The CNN model consists of the following layers:

Convolutional layer with 32 filters, kernel size of 3x3, ReLU activation
MaxPooling layer with pool size of 2x2
Convolutional layer with 64 filters, kernel size of 3x3, ReLU activation
MaxPooling layer with pool size of 2x2
Flatten layer
Fully connected layer with 128 units, ReLU activation
Output layer with 10 units (for 10 classes), softmax activation
Results
The model achieves high accuracy in classifying handwritten digits, with an accuracy of over 99% on the test set.

Acknowledgements
The MNIST dataset is provided by Yann LeCun and can be found here.
This project uses TensorFlow and Keras libraries for building and training the CNN model.
License
This project is licensed under the MIT License - see the LICENSE file for details.

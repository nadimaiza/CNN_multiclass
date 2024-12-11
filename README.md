# CNN_multiclass (88% accuracy):

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for multiclass classification using the FashionMNIST dataset. Three different model setups are explored, trained, and evaluated to predict fashion item classes, achieving a final accuracy of 88%.


Project Overview
This project focuses on building and experimenting with CNN architectures to classify the FashionMNIST dataset into its 10 predefined classes. The project workflow includes:

Data preparation and augmentation.
Implementation of three different CNN setups.
Training and validation with epoch-based loss tracking.
Hyperparameter tuning and optimization.
Achieving an 88% accuracy on the test set.
Dataset
The FashionMNIST dataset is a collection of grayscale images, each 28x28 pixels, representing 10 classes of fashion items:

T-shirts/tops
Trousers
Pullovers
Dresses
Coats
Sandals
Shirts
Sneakers
Bags
Ankle boots
Model Architectures
The project explores three different CNN architectures with varying complexities, including:

Baseline CNN: A simple architecture to establish a performance baseline.



Improved CNN: Includes additional convolutional layers and dropout for better generalization.



Advanced CNN: Incorporates more advanced techniques such as batch normalization and additional layers.









SETUP AND INSTALLATION:








To replicate this project, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/nadimaiza/CNN_multiclass.git
cd CNN_multiclass
Install the required Python packages:

bash
Copy code
pip install torch torchvision matplotlib
Run the notebook in your preferred environment (e.g., Jupyter Notebook or Google Colab).

Training and Evaluation
Key Steps:
Data Preparation:

Normalization and augmentation.

Splitting the data into training, validation, and test sets.
Training:

Models trained for multiple epochs.

Use of optimizers torch.optim.SGD and loss functions nn.CrossEntropyLoss().
Monitoring and visualizing training/validation loss.
Evaluation:

Performance measured using accuracy.

Confusion matrix and classification report for detailed analysis.
Commands:
Run the notebook to train and evaluate all three CNN setups:

bash
Copy code
python CNN_multiclass.ipynb
Results
The best-performing model achieved an accuracy of 88% on the test set. This demonstrates the potential of CNNs for fashion image classification.

Model	Test Accuracy
Baseline CNN	83%
Improved CNN	11%
Advanced CNN	88%

Special thanks to:

PyTorch for the deep learning framework.
The creators of the FashionMNIST dataset.

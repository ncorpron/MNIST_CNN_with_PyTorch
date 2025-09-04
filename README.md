🖤 MNIST CNN with PyTorch 🖤

Train a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch.
💻 Colab-friendly, supports GPU acceleration, and includes training, evaluation, and visualization.

📂 Project Structure
MNIST_CNN_PyTorch/
│
├── MNIST_CNN.ipynb        📝 Demo notebook for Colab
├── models.py              🏗 CNN architecture (SimpleCNN)
├── utils.py               ⚙️ Helper functions (e.g., set_seed)
├── train.py               🚀 Training workflow + checkpointing
├── evaluate.py            📊 Testing, confusion matrix, sample predictions
├── requirements.txt       📦 Python dependencies
├── LICENSE                📄 MIT License
├── README.md              📝 This file
├── models/                💾 Saved trained models
└── images/                🖼 Plots and sample prediction images

⚡ Requirements

Install dependencies:

pip install -r requirements.txt


Example requirements.txt:

torch
torchvision
matplotlib
numpy
seaborn
scikit-learn

🏋️ Usage
1️⃣ Training the Model
python train.py


Trains the CNN on the MNIST training set

Saves best model → models/mnist_best.pth ✅

Logs loss, training accuracy, and validation accuracy per epoch

2️⃣ Evaluating the Model
python evaluate.py


Loads models/mnist_best.pth

Computes test accuracy

Displays confusion matrix and sample predictions 🖼

3️⃣ Using the Notebook

Open MNIST_CNN.ipynb in Google Colab or locally

Run cells to visualize training/validation curves, confusion matrix, and sample predictions

Notebook imports modular scripts for easy reuse 🔄

📝 Notes

Fully reproducible with seeds for deterministic results 🎯

Modify train.py or evaluate.py for experiments without touching the notebook 🛠

Recommended: Use GPU in Colab for faster training ⚡

📄 License

This project is licensed under the MIT License
 🏷

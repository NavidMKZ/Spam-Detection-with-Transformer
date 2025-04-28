# Spam-Detection-with-Transformer

This project implements a deep learning model for SMS spam classification using a Transformer-based architecture. The model is trained on the "Ham vs Spam SMS Classification Dataset" and performs binary classification to identify whether a message is 'ham' (non-spam) or 'spam'. The model achieves approximately 100% accuracy on the training data and around 99% accuracy on the test data.

## Dataset

The dataset used in this project is the "Ham vs Spam SMS Classification Dataset." It contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. The dataset is sourced from Kaggle.

- **Dataset Link**:[Ham vs Spam SMS Classification Dataset on Kaggle](https://www.kaggle.com/datasets/hubashaikh/ham-vs-spam-sms-classification-dataset/data)

- **Access**: The dataset is loaded using KaggleHub for easy access:  
  ```python
  import kagglehub
  path = kagglehub.dataset_download("hubashaikh/ham-vs-spam-sms-classification-dataset")


---

## Model Components

### 1. Tokenization and Encoding
The input text is tokenized using the BERT tokenizer.

### 2. Transformer Model
A Transformer model is employed for classifying SMS messages. The model utilizes self-attention mechanisms for text classification tasks.

### 3. Feedforward Neural Network
In addition to the Transformer layers, the model contains a feedforward neural network (FNN) with three linear layers. Each linear layer is followed by the GELU activation function.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- KaggleHub
- NumPy
- Matplotlib (for visualization)
- scikit-learn
- pandas


## Results

The model achieves 100% accuracy on the training data and approximately 99% accuracy on the test data. The performance is evaluated using metrics like accuracy, F1 score, and a confusion matrix.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

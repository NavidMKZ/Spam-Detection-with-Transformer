# Spam-Detection-with-Transformer

This project implements a deep learning model for SMS spam classification using a Transformer-based architecture. The model is trained on the "Ham vs Spam SMS Classification Dataset" and performs binary classification to identify whether a message is 'ham' (non-spam) or 'spam'. The model achieves approximately 100% accuracy on the training data and around 99% accuracy on the test data.

## Dataset

The dataset used in this project is the "Ham vs Spam SMS Classification Dataset." It contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. The dataset is sourced from Kaggle.

- **Dataset Link**:[Ham vs Spam SMS Classification Dataset on Kaggle](https://www.kaggle.com/datasets/hubashaikh/ham-vs-spam-sms-classification-dataset/data)

- **Access**: The dataset is loaded using KaggleHub for easy access:  
  ```python
  import kagglehub
  data_path =  kagglehub.dataset_download('hubashaikh/ham-vs-spam-sms-classification-dataset')


---

## Model Components

### 1. Transformer Model
A Transformer model is employed for classifying SMS messages. The model utilizes self-attention mechanisms to capture long-range dependencies and important features from the text, making it effective for text classification tasks.

### 2. Tokenization and Encoding
The input text is tokenized using the BERT tokenizer, which converts each message into tokenized representations suitable for input to the Transformer model.

### 3. Focal Loss
A custom Focal Loss function is used to address the class imbalance between 'ham' and 'spam' messages. This loss function helps the model to focus more on difficult examples and improve classification performance.

Focal Loss Implementation: The custom implementation of Focal Loss is adapted from various sources and applied here to enhance the model's ability to classify imbalanced data.


## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- KaggleHub
- NumPy
- Matplotlib (for visualization)
- scikit-learn
- pandas
- tqdm


## Results

The model achieves 100% accuracy on the training data and approximately 99% accuracy on the test data. The performance is evaluated using metrics like accuracy, F1 score, and a confusion matrix.

Example Confusion Matrix:

![Confusion Matrix](images/confusion_matrix.png)

**Test Accuracy:** 99%  
**Test F1 Score:** 98.5%

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

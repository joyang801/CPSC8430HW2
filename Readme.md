https://github.com/joyang801/CPSC8430HW2

dataset.py: defines datsset for handling video caption datasets. It processes labels, builds a vocabulary and word indices for tokenization, and prepares the dataset for training or evaluation in machine learning tasks like video captioning. It includes methods for constructing the vocabulary from the training data and converting captions into tokenized form for use with a model.

model.py: defines the model for sequential data processing, and an embedding layer for handling vocabulary. The model is designed to generate sequences, such as captions, based on input features, with methods for both training and inference.

utils.py: defines functions and classes for evaluating a video captioning model. It includes functions to test the model, detokenize predictions, write results to a file, calculate BLEU scores for model evaluation, and plot training loss. 

configs.py: specifies paths to dataset resources and training hyperparameters for a machine learning model

train.ipynb: This code sets up and runs the training process for a sequence-to-sequence model on a video captioning task. It initializes the model, loads the training and testing datasets, and iterates through the epochs with progress tracking via `tqdm`. The model's performance is evaluated at specific intervals, and the training losses are plotted after completion. The trained model's parameters are saved for future use.

bleu_evaluation: This script sets up an environment for evaluating the sequence-to-sequence model's performance on a video captioning task. It loads the pretrained model and the test dataset, evaluates the model to generate captions for the test data, and calculates the BLEU score to assess the quality of the generated captions against the ground truth. The results are saved to an output file.

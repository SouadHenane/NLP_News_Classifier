from fastai import *
from fastai.text import *
import torch
import pandas as pd
import numpy as np
import re


def preprocessing(text):
    # clean_ascii
    results = "".join(i for i in text if ord(i) < 128)
    return results.lower()


class ModelPackageULMFIT:
    def __init__(self, weights_path, weights_name):
        """
        weights_path: path to the folder where sklearn model is saved
        weights_name: name of the file (model)
        """

        # Inference on CPU for fast.ai
        defaults.device = torch.device("cpu")
        # For Google Run, we only have 1CPU
        # defaults.cpus = 1

        # Load model
        try:
            # self.model = load_learner(weights_path, weights_name, num_workers=0)
            self.model = load_learner(weights_path, weights_name)
            self.model.model.eval()

        except IOError:
            print("Error Loading ULMfit Model")

        # class mapping to indices
        self.indice_2_class = {v: k for k, v in self.model.data.c2i.items()}

    def topk_predictions(self, text, k):
        """
        text: str, raw text input 
        k: int, to define top-k predictions

        returns a dict of k keys {classes: probability}
        """
        
        # Prepare input
        input_ = preprocessing(text)

        # Do inference, predictions[2] contains probabilities
        with torch.no_grad():
            predictions = self.model.predict(input_)

        # Select top_k predictions and their probabilities
        proba_k = predictions[2].topk(k)[0].data.numpy()
        indices_k = predictions[2].topk(k)[1].data.numpy()

        # dictionnary of predicted classes with their probabilities
        results = {
            self.indice_2_class[i]: "{:12.2f}%".format(float(j) * 100)
            for i, j in zip(indices_k, proba_k)
        }

        return results

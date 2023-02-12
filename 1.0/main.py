from trainer import trainer
from mlm_train import mlm_train
from dataset_builder import dataset_builder
from datasets import load_from_disk
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import cohen_kappa_score
def main(loadpath="../data/datasets/", epochs=5,
        savepath="../models/", overwrite=False):
    """Trains the a fine-tuned MLM for AES using cross validation
    Args:
        loadpath: directory for datasets
        epochs: number of epochs for training
        savepath: directory to save fold-models
        overwrite: set True to make new models even if previous exist
    """
    sets = [3,4,5,6,7] # Only using sets 3-7 since they fit within model size
    for essay_set in sets:
        model_savepath = savepath + "set" + str(essay_set) + "/"
        train_folds(essay_set, loadpath=loadpath, epochs=epochs,
                savepath=model_savepath)

def train_folds(set_no, folds=[0,1,2,3,4], loadpath="../data/datasets/",
        savepath="../models/", overwrite=False, epochs=5, batch_size=4):
    """Trains multiple models for cross fold validation, saves results
    args:
        set_no: current set to train
        folds: folds to train, can be modified to run a single fold
        loadpath: directory for fold datasets
        overwrite: set True to make new models even if previous exist
        epochs: number of epochs for training
        batch_size: batch size for training, modify if running out
    """
    predictions = []
    for fold in folds:
        print("Set: " + str(set_no) + ", fold: " + str(fold))
        filename = loadpath + "set" + str(set_no) + "/fold" + str(fold) +".data"
        print("Loading " + filename)
        if os.path.exists(filename) and overwrite == False:
            dataset = load_from_disk(filename)
            print("Dataset loaded from disk.")
        else:
            dataset_builder(set_no)
            dataset = load_from_disk(filename)
            print("Dataset created and saved.")
        # If trained model already exists
        m_file = savepath + "set" + str(set_no) + "/fold" + str(fold) + ".model"
        if os.path.exists(m_file):
            model = AutoModelForSequenceClassification.from_pretrained(m_file)
            print("Model loaded from disk.")
        else:
            model_trainer = trainer(dataset, m_file, epochs=epochs,
                    batch_size=batch_size)
            model = model_trainer.train()
            print("Model created and saved.")

        # Make predictions of test data essays
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        predictor = TextClassificationPipeline(
            model = model,
            tokenizer = tokenizer, #AutoTokenizer.from_pretrained("distilbert-base-cased"),
            top_k=1,
            device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")
            )

            # predictor(essay["text"]):
            # [{'label': 'LABEL1', 'score': 0.95}]
        expected_scores = [int(out[0]["label"][6:]) for out in predictor(
            KeyDataset(dataset["test"], "text"), batch_size=8, truncation=True)]
        true_scores = [int(label) for label in dataset["test"]["labels"]]
        essay_ids = [int(idx) for idx in dataset["test"]["idx"]]
        for i in range(len(expected_scores)):
            current = (essay_ids[i], expected_scores[i], true_scores[i])
            predictions.append(current)

        print("Fold " + str(fold) + " finished.")

    predictions.sort()
    df = pd.DataFrame(predictions, columns=["essay_id", "prediction",
        "true_score"])
    filename = loadpath + "set_" + str(set_no) + "_predictions.csv"
    df.to_csv(filename, index=False)
    print("Predictions saved to: " + filename)
    
    result_file = loadpath + "set_" + str(set_no) + "_results.txt"
    accuracy, kappa = analyze_predictions(df)
    with open(result_file, "w") as file:
        file.write("Set: " + str(set_no) + "\n")
        file.write("Accuracy: " + str(accuracy) + "\n")
        file.write("Kappa: " + str(kappa))
    print("results saved to: " + result_file)

            
def analyze_predictions(df):
    """Returns accuracy and cohen kappa for predictions
    """
    correct = 0
    total = len(df)
    for index, row in df.iterrows():
        if row["prediction"] == row["true_score"]:
            correct += 1
    accuracy = correct / total

    kappa = sklearn.metrics.cohen_kappa_score(
            df[["true_score"]],
            df[["prediction"]]
            )
    return (accuracy, kappa)
            
if __name__ == '__main__':
    main()

from trainer import trainer
from dataset_builder import dataset_builder
from datasets import load_from_disk
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
def train(loadpath="../data/datasets/", epochs=5,
        savepath="../models/set"):
    """
    """
    sets = [3,4,5,6,7]
    for essay_set in sets:
        model_savepath = savepath + str(essay_set) + "/"
        train_folds(essay_set, loadpath=loadpath, epochs=epochs,
                savepath=model_savepath)

def train_folds(set_no, folds=[0,1,2,3,4], loadpath="../data/datasets/",
        savepath="../models/", overwrite=False, epochs=5):
    """Trains multiple models for cross fold validation, print and saves results.
    Args:
        set_no: essay_set number
        folds: list of which folds to evaluate
        loadpath: path to dataset directory
        savepath: path to model save directory
        overwrite: if previously saved models should be replaced
        epochs: number of epochs to train model

    """
    predictions = []
    for fold in folds:
        print("Set: " + str(set_no) + ", fold: " + str(fold))
        filename = loadpath + "set" + str(set_no) + "/fold" + str(fold) + ".data"
        print("Loading " + filename)
        print("Filepath found: " + str(os.path.exists(filename)))
        if os.path.exists(filename):
            dataset = load_from_disk(filename)
            print("Dataset loaded!")
        else:
            dataset_builder(set_no)
            dataset = load_from_disk(filename)
            print("Dataset created and loaded!")
        
        test_data = dataset["test"]
        model_trainer = trainer(dataset, epochs=epochs,
                checkpoint="../models/essay_model.model")
        new_predictions = model_trainer.train()
        predictions = predicitons + new_predictions
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
        file.write("Accuracy: " + str(accuracy))
        file.write("Kappa: " + str(kappa))
    print("results saved to: " + result_file)

def analyze_predictions(df):
    correct = 0
    total = len(df)
    for index, row in df.iterrows():
        if row["prediction"] == row["true_result"]:
            correct += 1
    accuracy = correct / total

    kappa = sklearn.metrics.cohen_kappa_score(
            df[["true_result"]],
            df[["prediction"]]
            )
    return (accuracy, kappa)
            
if __name__ == '__main__':
    # Run fold_train from command line
    train()

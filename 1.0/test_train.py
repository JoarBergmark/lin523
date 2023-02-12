6rom trainer import trainer
from dataset_builder import dataset_builder
from datasets import load_from_disk
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline
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
        savepath="../models/", overwrite=False, epochs=5, batch_size=4):
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
        filename = loadpath + "set" + str(set_no) + "/fold" + str(fold)
        print("Loading " + filename + ".data")
        print("Filepath found: " + str(os.path.exists(filename + ".data")))
        if os.path.exists(filename + ".data"):
            dataset = load_from_disk(filename + ".data")
            print("Dataset loaded!")
        else:
            dataset_builder(set_no)
            dataset = load_from_disk(filename + ".data")
            print("Dataset created and loaded!")
        
        filename = filename + ".model"
        if os.path.exists(filename):
            model = AutoModelForSequenceClassification.from_pretrained(filename)
            print("Model loaded from disk.")
        else:
            model_trainer = trainer(dataset, filename, epochs=epochs,
                    model_save=filename, batch_size=batch_size)
            model = model_trainer.train()
            print("Model created and saved.")

        # Make predictions of test data essays
        predictor = TextClassificationPipeline(
            model = model,
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased"),
            top_k=1)
        for essay in dataset["test"]:
            essay_id = essay["idx"]
            print("**EXPECTED_SCORE**:")
            print(predictor(essay["text"]))
            print(predictor(essay["text"])[0][0]["label"])
            print(predictor(essay["text"])[0][0]["label"][5:])
            # predictor(essay["text"]):
            # [[{'label': 'LABEL1', 'score': 0.95}]]
            print(essay["labels"])

            expected_score = int(predictor(essay["text"])[0][0]["label"][5:])
            true_score = essay["labels"]  
            predictions.append(tuple(essay_id, expected_score, true_score))

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

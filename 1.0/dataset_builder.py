import datasets
from datasets import Dataset, DatasetDict, load_dataset, Features, ClassLabel
import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def dataset_builder(set_no, path="../data/training_set_rel3.tsv",
        savepath="../data/datasets/", folded=5):
    """Loads data from .tsv-file and saves a DatasetDict object with the
    specified set of essays.
    Args:
        path: path to .tsv-file locations
        savepath: path to save the dataset objects

    """
    set_dataset = data_from_csv(set_no, path)
    set_dataset.shuffle()
    print("Data loaded")
    
    kfolds = create_folds(set_dataset)
    for num, folds in enumerate(kfolds):
        save_dir = savepath + "set" + str(set_no) + "/fold" + str(num) + ".data"
        folds.save_to_disk(save_dir)
        print("Dataset saved to: " + save_dir)
    print("Dataset folds saved.")

def create_folds(dataset, k=5):
    """Takes a dataset and returns k cross fold validation splits.
    Args:
        dataset: dataset object containing all items
        k: number of folds
    The folds are stratified, meaning they will have an equal division of
    "labels" values.
    """
    folds = StratifiedKFold(k)
    # Make splits stratified based on labels
    splits = folds.split(np.zeros(dataset.num_rows), dataset["labels"]) 
    fold_list = list()

    for train_idxs, test_idxs in splits:
        # Creates indexes for splitting training split into train and validation
        train_ds = dataset.select(train_idxs)
        skf = StratifiedKFold(k)
        # Makes the validation set split stratified
        tv_splits = skf.split(np.zeros(train_ds.num_rows), train_ds["labels"])
        for train, val in tv_splits:
        # Final splits are 20% test, 80% test+validation (split 80/20 again)
            fold_dataDict = DatasetDict({
                "train": train_ds.select(train),
                "validation": train_ds.select(val),
                "test": dataset.select(test_idxs)
                })
            # Appends current dataDict to folds.
            fold_list.append(fold_dataDict)
            # Breaks loop to avoid creating unessecary folds.
            break
    return fold_list

def data_from_csv(set_no, filename="../data/training_set_rel3.tsv"):
    """Extracts relevant rows and columns to a dataset object.
    Args:
        filename: path to .tsv-file
        set_no: the essay_set number
    """
    # Read .csv to pandas dataframe
    df = pd.read_csv(filename, sep="\t", encoding="ISO-8859-1")
    df = df[df["essay_set"] == set_no]
    df = df[["essay", "domain1_score", "essay_id"]]
    df = df.rename(columns={
        "essay": "text",
        "domain1_score": "labels",
        "essay_id": "idx"
        })
    # Decrement label for set 1, 2 to avoid label > num_labels in ClassLabel
    if set_no == 1:
        df["labels"] = df["labels"] - 2
    elif set_no == 2:
        df["labels"] = df["labels"] - 1

    current_data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    # set_info used for creation of ClassLabel for each set.
    set_info = {
            1: range(2, 13),
            2: range(1, 7),
            3: range(0, 4),
            4: range(0, 4),
            5: range(0, 5),
            6: range(0, 5),
            7: range(0, 31),
            8: range(0, 61)
            }
    this_set = set_info[set_no]        
    current_data = current_data.cast_column("labels",
            ClassLabel(
                num_classes=len(this_set),
                names=list(map(str, this_set))
                )
            )
    #print(current_data.features)
    #sample = current_data.shuffle()
    #print(sample[:5]["labels"])
    return current_data


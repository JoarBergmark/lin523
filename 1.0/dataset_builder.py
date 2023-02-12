import datasets
from datasets import Dataset, DatasetDict, load_dataset, Features, ClassLabel,\
load_from_disk
import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def dataset_builder(set_no, path="../data/training_set_rel3.tsv",
        savepath="../data/datasets/", folded=5):
    """Loads data from .tsv-file and saves a DatasetDict object with the
    specified set of essays
    Args:
        path: path to .tsv-file locations
        savepath: path to save the dataset objects
        folded: number of folds (default 5)
    """
    set_dataset = data_from_csv(set_no, path)
    set_dataset.shuffle()
    if folded > 1: 
        kfolds = create_folds(set_dataset)
        for num, folds in enumerate(kfolds):
            save_dir = savepath + "set" + str(set_no) + "/fold" + str(num) + ".data"
            folds.save_to_disk(save_dir)
            print("Dataset saved to: " + save_dir)
        print("Dataset folds saved.")
    elif folded == 1:
        tempDict = set_dataset.train_test_split(test_size=0.2, shuffle=True)
        Dict = tempDict["train"].train_test_split(test_size=0.2) 
        Dict["validation"] = Dict.pop("test")
        Dict["test"] = tempDict["test"]
        save_dir = savepath + "set" + str(set_no) + ".data"
        Dict.save_to_disk(save_dir)
        print("Dataset saved to: " + save_dir)

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

def all_essays_dataset(path="../data/",
        savepath="../data/datasets/all_essays.data"):
    """Returns dataset of all essays, builds and saves dataset in none exists.
    Args:
        path: path to saved .tsv files
        savepath: savepath for dataset
    """
    if os.path.exists(savepath):
        dataset = load_from_disk(savepath)
        return dataset
    essay_df = pd.DataFrame() 
    for filename in os.listdir(path):
        if filename.endswith(".tsv"):
            filepath = os.path.join(path, filename)
            df = pd.read_csv(filepath, sep="\t", encoding="ISO-8859-1")
            essay_df = pd.concat((essay_df, df[["essay", "essay_id"]]),
                    ignore_index=True)
    essay_df.sort_values(by="essay_id")
    essay_df = essay_df[["essay"]]
    dataset = Dataset.from_pandas(essay_df)
    dataset.save_to_disk(savepath)
    return dataset

def dataset_unique_chars(path="../data/"):
    """Returns, and saves, a list of unique characters used in all_essays_dataset
    """
    unique_characters = set()

    essay_df = pd.DataFrame() 
    for filename in os.listdir(path):
        if filename.endswith(".tsv"):
            filepath = os.path.join(path, filename)
            df = pd.read_csv(filepath, sep="\t", encoding="ISO-8859-1")
            essay_df = pd.concat((essay_df, df[["essay", "essay_id"]]),
                    ignore_index=True)
    essay_df.sort_values(by="essay_id")
    essay_df = essay_df[["essay"]]
    print(len(essay_df))
    all_text = "".join(essay_df["essay"])
    unique_chars = list(set(all_text))
    unique_chars.sort()

    print(unique_chars)
    with open("../info/unique_chars.txt", "w", encoding="ISO-8859-1") as file:
        for char in unique_chars:
            file.write(char + "\n")

def replace_characters(example):
    """Replaces unusual characters with common counterparts, returns text+count
    Args:
        example: string
    """
    output_text = example
    no_replacements = Counter()
    codes_to_chars = {
        "": "€",
        "": "...",
        "": "\'",
        "": "\'",
        "": "\"",
        "": "\"",
        "": "-",
        "": "<",
        "": ">",
        "": "oe",
        "": "",
        "­": "-", #The first hyphen is a different character.
        }
    for code in codes_to_chars:
        no_replacements["[" +code + "] / [" + codes_to_chars[code] + "]"] =\
            output_text.count(code)
        if no_replacements[code + "/" + codes_to_chars[code]] > 0:
            output_text = output_text.replace(code, codes_to_chars[code])
    return (output_text, no_replacements)

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
    # Replace uncommon characters with common ones and print replacements
    replacement_counts = Counter()
    for text in df["text"]:
        update = replace_characters(text)
        text = update[0]
        replacement_counts = replacement_counts + update[1]
    print(replacement_counts)
 
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
    return current_data

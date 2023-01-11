import datasets
from datasets import load_dataset
from datasets import Dataset
import glob
import os
import pandas as pd

def dataload(path, savepath, set_no):
    """Loads data from .tsv-file and saves a DatasetDict object with the
    specified set of essays.
    Args:
        path: path to .tsv-file locations
        savepath: path to save the dataset objects

    """
    data_files = {
            "train": (data_from_csv(path + "training_set_rel3.tsv", set_no)),
            "validation": (data_from_csv(path + "valid_set.tsv", set_no)),
            "test": (data_from_csv(path + "test_set.tsv", set_no))
            }
    dataDict = datasets.DatasetDict(data_files)
    print(dataDict)
    print("Dataset loaded!")
    # print(dataDict["train"].features)
    #
    # {"sequence": Value(dtype="string", id=None), "labels": Value(dtype="int64", 
    # id=None))
    # }

    save_dir = savepath + "/" + str(set_no)
    dataDict.save_to_disk(save_dir)
    print("Dataset saved to " + save_dir + " as " + str(set_no))
    

    #for split, dataset in dataDict.items():
    #    dataset.to_csv(f"essay_set-{set_no}-{split}.csv", index=None)

def data_from_csv(filename, set_no):
    """Extracts relevant rows and columns to a dataset object.
    Args:
        filename: path to .tsv-file
        set_no: the essay_set number
    """
    # Read .csv to pandas dataframe
    df = pd.read_csv(filename, sep="\t", encoding="ISO-8859-1")
    df = df[df["essay_set"] == set_no]
    if "domain1_score" in df.columns:
        df = df[["essay", "domain1_score"]]
        df = df.rename(columns={"essay": "sequence", "domain1_score": "labels"})
    else:
        df = df[["essay"]]
        df = df.rename(columns={"essay": "sequence"})
            
    current_data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    return current_data


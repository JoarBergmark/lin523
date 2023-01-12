import datasets
from datasets import Dataset, load_dataset, Features, ClassLabel
import glob
import os
import pandas as pd

def dataset_builder(path, savepath, set_no):
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
    elif "valid_set" in filename:
        # Open validation set scores from same path
        score_file = (filename[:-len("valid_set.tsv")] +
                "valid_sample_submission_2_column.csv")
        df2 = pd.read_csv(score_file, sep=",")
        df2 = df2.rename(columns={"prediction_id": "essay_id",
            "predicted_score": "labels"}
            )
        df = pd.merge(df, df2[["essay_id", "labels"]], on="essay_id")
        df = df[["essay", "labels"]]
        df = df.rename(columns={"essay": "sequence"})
    else:
        df = df[["essay"]]
        df = df.rename(columns={"essay": "sequence"})
        df = df.assign(labels = -1)

    current_data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    # Makes a ClassLabel object with info about the labels, relative to set_no
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

    print(current_data.features)
    #sample = current_data.shuffle()
    #print(sample[:3])
    #print(sample.features)
    return current_data



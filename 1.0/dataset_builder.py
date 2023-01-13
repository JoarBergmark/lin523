import datasets
from datasets import Dataset, load_dataset, Features, ClassLabel
import glob
import os
import pandas as pd

def dataset_builder(set_no, path="../data/training_set_rel3.tsv", savepath="../data/datasets/"):
    """Loads data from .tsv-file and saves a DatasetDict object with the
    specified set of essays.
    Args:
        path: path to .tsv-file locations
        savepath: path to save the dataset objects

    """
    set_dataset = data_from_csv(set_no, path)
    set_dataset.shuffle()
    
    # Split original training set files into train and test sets.
    data_dict = set_dataset.train_test_split(test_size=0.2)
    # Splits train into an additional validation set
    data_dict["train"], data_dict["validation"] = \
            [data_dict["train"].train_test_split(test_size=0.2)[i]\
                for i in ("train","test")]
    print("Dataset loaded!")
    print(data_dict)

    save_dir = savepath + str(set_no) + ".data"
    data_dict.save_to_disk(save_dir)
    print("Dataset saved to " + save_dir + " as " + str(set_no) + ".data")
    
def data_from_csv(set_no, filename="../data/training_set_rel3.tsv"):
    """Extracts relevant rows and columns to a dataset object.
    Args:
        filename: path to .tsv-file
        set_no: the essay_set number
    """
    # Read .csv to pandas dataframe
    df = pd.read_csv(filename, sep="\t", encoding="ISO-8859-1")
    df = df[df["essay_set"] == set_no]
    df = df[["essay", "domain1_score"]]
    df = df.rename(columns={"essay": "sequence", "domain1_score": "labels"})
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


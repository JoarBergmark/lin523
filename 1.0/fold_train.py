from gpu_train_model import train_model
from dataset_builder import dataset_builder
from datasets import load_from_disk

def train(set_no, folds=[0,1,2,3,4], loadpath="../data/datasets/",
        savepath="../models/", overwrite=False, epochs=3):
    """Trains multiple models for cross fold validation, print and saves results.
    Args:
        set_no: essay_set number
        folds: list of which folds to evaluate
        loadpath: path to dataset directory
        savepath: path to model save directory
        overwrite: if previously saved models should be replaced
        epochs: number of epochs to train model

    """
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
                
        if False:#model_file_exists:
            print()
        else:
            name = "set" + str(set_no) + "_fold" + str(fold) + ".model"
            train_model(dataset, (savepath + name), n_epochs=epochs)

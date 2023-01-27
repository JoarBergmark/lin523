from trainer import trainer
from dataset_builder import dataset_builder
from datasets import load_from_disk
import os
import sys
def train(set_no, loadpath="../data/datasets/", epochs=5):
    filename = loadpath + "set" + str(set_no) + ".data"
    print("filename: " + filename)
    dataset = load_from_disk(filename)
    model_trainer = trainer(dataset, epochs)
    model_trainer.train()
    print("Training finished in test_train")

if __name__ == '__main__':
    # Run fold_train from command line
    print(sys.argv) 
    #args =  for arg in len(sys.argv) - 1
    if sys.argv[1] == "train":
        train(sys.argv[2])

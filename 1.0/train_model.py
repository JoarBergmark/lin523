from load_model import load_pretrained
from dataset_builder import dataset_builder
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import pandas as pd
import evaluate
import os
def train_model(set_no, dataset_path="../data/datasets/", savepath="../models/"):
    """Loads a model, defines training parameters and datasets.
    Args:
        savepath: directory to save trained model
        set_no: essay set for model
    """
    filename = dataset_path + str(set_no) + ".data"
    print("Loading " + filename)
    print("Filepath found: " + str(os.path.exists(filename)))
    if os.path.exists(filename):
        dataset = load_from_disk(filename)
        print("Dataset loaded!")
    else:
        dataset_builder(set_no)
        dataset = load_from_disk(filename)
        print("Dataset created and loaded")

    # Load pretrained model and tokenizer
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize function for dataset.map()
    def tokenize_function(essay):
        return tokenizer(essay["sequence"], truncation=True)

    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy", "precision")
            # labels borde vara "orden" som motsvarar kategorier
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, refrences=labels)    

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    #Tokenized dataset features:
    #    sequence: Value(string)
    #    labels: ClassLabel
    #    input_ids: Sequence(Value)
    #    token_type_ids: Sequence(Value)
    #    attention_mask: Sequence(Value)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    savefile = savepath + str(set_no) + ".model"
    training_args = TrainingArguments(
            savepath,
            evaluation_strategy="epoch"
            )

    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=(dataset["train"].features["labels"].num_classes)
            )
    
    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator, # denna rad behövs inte för detta
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
            )
    #Detta är nog redan inbyggt i compute_metrics()
        # predictions of validation set
    #predictions = trainer.predict(tokenized_dataset["validation"])
        # highest prediction of each validation set essay
    #preds = np.argmax(predictions.predictions, axis=-1)
    
    print("Training starts here!")
    #trainer.train()



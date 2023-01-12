from load_model import load_pretrained
from dataset_builder import dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import pandas as pd
import evaluate
import os
def train_model(dataset_path="../data/datasets/", savepath="../models", set_no):
    """Loads a model, defines training parameters and datasets.
    Args:
        savepath: directory to save trained model
        set_no: essay set for model
    """
    
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path + str(set_no))
        print("Dataset loaded!")
    else:
        dataset_builder("../data/", dataset_path, set_no)
        dataset = load_dataset(dataset_path + str(set_no))
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
            savepath,
            evaluation_strategy="epoch"
            )

    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=(dataset.features["labels"]["num_classes"])
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

    trainer.train()



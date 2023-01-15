from dataset_builder import dataset_builder
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import pandas as pd
import evaluate
import os
from pynvml import *
import torch
import sklearn
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

        print(eval_preds)
        print(logits)
        print(labels)

        # Old code that gets me NoneType error
        #predictions = np.argmax(logits, axis=-1)
        #return metric.compute(predictions=predictions, refrences=labels)    
        
        # If model returns more than the preiction logits
        if isinstance(logits, tuple):
            print("*************IT WAS AN INSTANCE OF LOGITS, TUPLE!***************")
            quit()
            logits = logits[0]
        decoded_logits = tokenizer.batch_decode(logits, skip_special_tokens=True)
        # Replace "-100" in labels since they cant decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-proccesing
        decoded_logits = [logit.strip() for logit in decoded_logits]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        return metric.compute(predictions=decoded_logits,
                refrences=decoded_labels)
        

    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

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
            evaluation_strategy="epoch",
            gradient_accumulation_steps=4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            )
    print("GPU memory before model: ")
    print_gpu_utilization()
    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=(dataset["train"].features["labels"].num_classes)
            ).to("cuda")
    print("GPU memory with loaded model: ")
    print_gpu_utilization()
    
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
    print("GPU memory at start: ")
    print_gpu_utilization()
    result = trainer.train()
    print_summary(result)


from dataset_builder import dataset_builder
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, AdamW, get_scheduler
import numpy as np
import pandas as pd
import evaluate
import os
from pynvml import *
import torch
from torch.utils.data import DataLoader
import sklearn
def train_model(dataset, savepath, n_epochs=3):
    """Loads a model, defines training parameters and datasets.
    Args:
        savepath: directory to save trained model
        set_no: essay set for model
    """
    #filename = dataset_path + str(set_no) + ".data"
    #print("Loading " + filename)
    #print("Filepath found: " + str(os.path.exists(filename)))
    #if os.path.exists(filename):
    #    dataset = load_from_disk(filename)
    #    print("Dataset loaded!")
    #else:
    #    dataset_builder(set_no)
    #    dataset = load_from_disk(filename)
    #    print("Dataset created and loaded")

    # Load pretrained model and tokenizer
    checkpoint = "distilbert-base-uncased"
    #checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize function for dataset.map()
    def tokenize_function(essay):
        return tokenizer(essay["text"], truncation=True)

    def compute_metrics(eval_preds):
        metric1 = evaluate.load("f1")
        metric2 = evaluate.load("accuracy")
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        m1 = metric1.compute(predictions=preds, references=labels, average="micro")
        m2 = metric2.compute(predictions=preds, references=labels)
        return {"f1": m1, "accuracy": m2}

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
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "idx"])
#tokenized_datasets features:{
#'labels':ClassLabel(names=['0', '1', '2', '3'],id=None),
#'input_ids':Sequence(feature=Value(dtype='int32',id=None),length=-1,id=None),
#'token_type_ids':Sequence(feature=Value(dtype='int8',id=None),length=-1,id=None),
#'attention_mask':Sequence(feature=Value(dtype='int8',id=None),length=-1,id=None)}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
            savepath,
            num_train_epochs=n_epochs,
            evaluation_strategy="epoch",
            gradient_accumulation_steps=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            )
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("GPU memory before model: ")
    print_gpu_utilization()
    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=(dataset["train"].features["labels"].num_classes)
            ).to("cuda")
    print("GPU memory with loaded model: ")
    print_gpu_utilization()
    
    
    train_dataloader = DataLoader(
            tokenized_dataset["train"], batch_size=4, collate_fn=data_collator
            )
    eval_dataloader = DataLoader(
            tokenized_dataset["validation"], batch_size=4,
            collate_fn=data_collator
            )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(n_epochs * len(train_dataloader))
            )

    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator, # denna rad behövs inte för detta
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            ) 
    #dataloader = trainer.get_train_dataloader()
    #trainer.create_optimizer_and_scheduler(n_epochs * len(dataloader))
    #print("Trainer got optimizer.")
    #quit()

    print("Training starts here!")
    print("GPU memory at start: ")
    print_gpu_utilization()
    result = trainer.train()
    print_summary(result)
    trainer.save_model(savepath)

    #print("Model trained, results for test split: ")
    #print(compute_metrics(trainer.predict(tokenized_dataset["test"])))



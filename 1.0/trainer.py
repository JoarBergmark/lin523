from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TextClassificationPipeline
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm # For progress bar
import evaluate
import gc

# This follows the HuggingFace pytorch trainer tutorial:
# https://huggingface.co/course/chapter3/4?fw=pt
class trainer(object):
    """Trainer class for model initiation and training.
    """
    # this was created as for the same trainer object to be used for every fold
    # training, but wierd memory issues had me make a new trainer for each
    # instance.
    def __init__(self, dataset, savepath, model_save="../models/essay_mlm.model",
            checkpoint="distilbert-base-cased", epochs=5, batch_size=4):
        """
        Args:
            dataset: dataset for training
            savepath: savepath for trained model
            model_save: directory for fine-tuned MLM
            checkpoint: base model, used for retriewing tokenizer
            epochs: number of epochs for training
            batch_size: number of essays per batch
        """
        self.dataset = dataset
        self.savepath = savepath
        self.model_save = model_save
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available()\
                else torch.device("cpu")
        print("torch.device = " + str(self.device))
    def tokenize_function(self, essay):
        """tokenize function for dataset.map()
        """
        return self.tokenizer(essay["text"], truncation=True)

    def train(self):
        """Trains a model and saves trained model
        """
        tokenized_datasets = self.dataset.map(self.tokenize_function,
                )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "idx"])
        tokenized_datasets.set_format("torch")

        train_dataloader = DataLoader(
                tokenized_datasets["train"],
                shuffle=True,
                batch_size=self.batch_size,
                collate_fn=data_collator
                )
        eval_dataloader = DataLoader(
                tokenized_datasets["validation"],
                shuffle=True,
                batch_size=self.batch_size, 
                collate_fn=data_collator
                )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_save,
            num_labels=(self.dataset["train"].features["labels"].num_classes)
            )
        gc.collect()
        torch.cuda.empty_cache()
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        #optimizer = torch.optim.AdamW(model.parameters)

        num_training_steps = self.epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
                )
        print("print(num_training_steps): " + str(num_training_steps))

        progress_bar = tqdm(range(num_training_steps))
        print("Training start:")
        # Set model to training mode
        model.train()
        for epoch in range(self.epochs):
            batch_losses = list()
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                batch_losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            epoch_loss = sum(batch_losses) / len(batch_losses)
            print("\n Training loss: " + str(epoch_loss))
            print("\n Epoch " + str(epoch + 1) + " evaluation: ")
            print(self.evaluate(model, eval_dataloader))
            print("\n")

        print("Training Finished!")
        model.save_pretrained(self.savepath)
        return model 

    def evaluate(self, model, eval_dataloader):
        """Gets evaluations for each epoch
        """
        metric1 = evaluate.load("f1")
        metric2 = evaluate.load("accuracy")
        # Set model to evaluation mode
        model.eval()
        batch_losses = list()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            batch_losses.append(outputs.loss.item())
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            metric1.add_batch(predictions=preds, references=batch["labels"])
            metric2.add_batch(predictions=preds, references=batch["labels"])
        eval_loss = sum(batch_losses) / len(batch_losses)
        # Set model back to training mode
        model.train()
        evals = metric1.compute(average="micro")
        evals.update(metric2.compute())
        evals.update({"eval_loss": eval_loss})
        return evals


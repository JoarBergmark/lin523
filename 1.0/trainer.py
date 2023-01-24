from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm # For progress bar
import evaluate
class trainer(object):
    """Trainer class for model initiation and training.
    """
    def __init__(self, dataset, checkpoint="distilbert-base-uncased", epochs=3):
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.epochs = epochs

    def tokenize_function(self, essay):
        return self.tokenizer(essay["text"], truncation=True)

    def train(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function,
                batched=True
                )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "idx"])
        tokenized_datasets.set_format("torch")

        #print(tokenized_datasets["train"].column_names)
        # ["labels", "input_ids", "attention_mask"]
        
        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True,
                batch_size=8, collate_fn=data_collator
                )
        eval_dataloader = DataLoader(tokenized_datasets["validation"],
                shuffle=True, batch_size=8, collate_fn=data_collator
                )

        # batch structure:
        # {'labels': torch.Size([8]), 'input_ids': torch.Size([8, 247]),
        # 'attention_mask': torch.Size([8, 247])}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=(self.dataset["train"].features["labels"].num_classes)
            )
        

        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_training_steps = self.epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
                )
        print("print(num_training_steps): ")
        print(num_training_steps)
        
        device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        model.to(device)
        progress_bar = tqdm(range(num_training_steps))
        print("Training start:")
        # Set model to training mode
        model.train()
        for epoch in range(self.epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            print("Epoch + " + str(epoch) + " evaluation: ")
            print(self.evaluate(model, eval_dataloader))

        print("Training Finished!")
        print("Saving model not implemented.")

    def evaluate(self, model, eval_dataloader):
        metric1 = evaluate.load("f1")
        metric2 = evaluate.load("accuracy")
        # Set model to evaluation mode
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            metric1.add_batch(predictions=preds, references=batch["labels"])
            metric2.add_batch(predictions=preds, references=batch["labels"])
        model.train()
        f1 = metric1.compute(avrage="micro")
        accuracy = metric2.compute()
        return {"f1": f1, "accuracy": accuracy}

    #def compute_metrics(eval_preds):
    #    metric1 = evaluate.load("f1")
    #    metric2 = evaluate.load("accuracy")
    #    logits, labels = eval_preds
    #    preds = np.argmax(logits, axis=-1)
    #    m1 = metric1.compute(predictions=preds, references=labels, average="micro")
    #    m2 = metric2.compute(predictions=preds, references=labels)
    #    return {"f1": m1, "accuracy": m2}

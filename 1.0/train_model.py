from load_model import load_pretrained
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import pandas as pd
import evaluate
def train_model(savepath, set_no):
    """
    """
    dataset = load_dataset()
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(essay):
        return tokenizer(essay["sequence"], truncation=True)

    def get_validation_scores():
        """Retrieves the validation scores for selected essay_set.
        Args:
            set_no: the essay_set number
        """
        # Read .csv to pandas dataframe
        filepath = "../data/valid_sample_submission_5_column.csv"
        df = pd.read_csv(filepath, sep=",", encoding="ISO-8859-1")
        df = df[df["essay_set"] == set_no]
        df = df[["predicted_score"]]
        scores = df.to_numpy(dtype=int) 
        return scores

    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy", "precision")
            # labels borde vara "orden" som motsvarar kategorier
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        true_scores = get_validation_scores(set_no)
        return metric.compute(predictions=predictions, refrences=true_scores)

    


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingARguments(
            "test-trainer",
            evaluation_strategy="epoch"
            )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
            num_labels=len(set(dataset["train"]["labels"])))
    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator, # denna rad behövs inte för detta
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
            )
    
    # predictions of validation set
    predictions = trainer.predict(tokenized_dataset["validation"])
    

    # highest prediction of each validation set essay
    preds = np.argmax(predictions.predictions, axis=-1)

    trainer.train()



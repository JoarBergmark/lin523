from dataset_builder import all_essays_dataset 
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM, AutoTokenizer,\
DataCollatorForLanguageModeling, default_data_collator
from datasets import load_from_disk
import torch
import collections
import numpy as np
import math
def mlm_train(checkpoint="distilbert-base-uncased",
        loadpath="../data/datasets/all_essays.data",
        savepath="../models/essay_mlm.model",):
    """
    Args:
        
    """
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Kanske mindre för att spara på minnet
    chunk_size = 128 # tokenizer.model_max_length = 512
    
    raw_dataset = all_essays_dataset()

    #Sample dataset for testing:
    #raw_dataset = raw_dataset.shuffle(seed=42).select(range(4000))
    #print(raw_dataset)

    dataset = raw_dataset.train_test_split(test_size=0.1)

    def tokenize_function(examples):
        """Tokenize funciton for dataset.map()
        """
        result = tokenizer(examples["essay"])
        if tokenizer.is_fast:
            # Add word_ids to preserve words split by tokenizer
            result["word_ids"] = [result.word_ids(i) for i in
                range(len(result["input_ids"]))]
        return result

    def group_texts(examples):
        """Groups tokenized essays into chunks of chunk_length
        """
        # Concatenate all the essays
        concatenated_examples = {k: sum(examples[k], []) for k in
                examples.keys()}
        # Length of concatenated essays
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop last chunk if smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of total_length
        result = {
                k: [t[i: i + chunk_size] for i in range(0, total_length,
                    chunk_size)] for k, t in concatenated_examples.items()
                }
        # Labels will be the "correct results" for masked token predictions
        result["labels"] = result["input_ids"].copy()
        return result
 
    def whole_word_masking_data_collator(features, wwm_probability=0.2):
        """Data collator for whole word masking MLM.
        Args:
            features:
            wwm_probability: share of words to be masked (default 20%)
        """
        for feature in features:
            word_ids = feature.pop("word_ids")
            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)
            # Mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping)))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels
        return default_data_collator(features)

    tokenized_datasets = dataset.map(tokenize_function, batched=True,
            remove_columns=["essay"])

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    
    #samples = [lm_datasets["train"][i] for i in range(1)]
    #batch = whole_word_masking_data_collator(samples)
    #for chunk in batch["input_ids"]:
    #    print(f"\n'>>> {tokenizer.decode(chunk)}'")
    #
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
    #        mlm_probability=0.2)
    #for sample in samples:
    #    _ = sample.pop("word_ids")
    #for chunk in data_collator(samples)["input_ids"]:
    #    print(f"\n'>>>{tokenizer.decode(chunk)}'")
    
    #Trainer API
    batch_size = 16
    # Report training loss every epoch
    logging_steps = len(lm_datasets["train"]) // batch_size
    # model_name = name for pushing to hub?
    training_args = TrainingArguments(
            output_dir=savepath,
            overwrite_output_dir=True,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            num_train_epochs=3,
            logging_steps=logging_steps,
            remove_unused_columns=False, # To save word_ids column
            #Maybe for saving along the way: save_steps=5000,
            #Maybe for memory: eval_accumulation_steps=2,
            )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
            data_collator=whole_word_masking_data_collator,
            tokenizer=tokenizer,
            )

    eval_results = trainer.evaluate()
    print("Perplexity: ")
    perplex = math.exp(eval_results["eval_loss"])
    print(perplex)

    trainer.train()

    eval_results = trainer.evaluate()
    print("Perplexity before training:")
    print(perplex)
    print("Perplexity after training: ")
    print(math.exp(eval_results["eval_loss"]))
    
    trainer.save_model(savepath)
    print("Model saved to: " + savepath)
    
if __name__ == '__main__':
    # Run from command line
    mlm_train()

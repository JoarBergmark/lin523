from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import BertConfig, BertModel
from datasets import load_from_disk
def load_pretrained(loadpath, savepath, set_no):
    """Loads a pretrained BERT-model from the hub.
    """
    # For sets with more than 300 avrage words, more than 512 token embeddings:
    #if set_no in [1, 2, 8]:
    #    bert_config = BertConfig.from_pretrained("bert-base-cased",
    #            max_position_embeddings=1024)
    #    bert_model = BertModel(bert_config)
    
    checkpoint = "bert-base-cased"
    # Load tokenizer from hub
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Pretrained model from hub
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    dataset_path = loadpath + str(set_no)
    datasets = load_from_disk(dataset_path)
    
    # dataset structure:
    # data_item {
    #   "sequence": (str) "essay text here"  
    #   "label": (int) domain1_score
    #    }

    def tokenize_function(essay):
        return tokenizer(essay["sequence"], truncation=True)

    sample = datasets["train"].shuffle(seed=42).select(range(2))
    print("\n Samples:")
    print(sample["sequence"])
    sample.map(tokenize_function, batched=True)
    inputs = tokenizer(sample["sequence"])
    print("\n Inputs:")
    print(inputs["input_ids"])
    for token_essays in inputs["input_ids"]:
        print("---")
        print(tokenizer.convert_ids_to_tokens(token_essays))

    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sequence"])
    tokenized_datasets = tokenized_datasets.with_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataloader = DataLoader(
            tokenized_datasets["train"], batch_size=32, shuffle=True,
            collate_fn=data_collator
            )
    
    for step, batch in enumerate(train_dataloader):
        print(batch["input_ids"].shape)
        if step > 5:
            break


from mlm_train import mlm_train
from transformers import AutoModel, LoadFromDisk
import trainer
import os
def main(overwrite=False, pretrained_mlm="../models/essay_mlm.model")
    """Runs model training and evaluation
    Args:
        overwrite:
        """
    if os.path.exists(pretrained_mlm) and overwrite=False:
        print("Loading pretrained essay model:")
        model = AutoModel.from_pretrained(pretrained_mlm)
    else:
        print("Pretrained model not found. Training essay_mlm.model.")
        mlm_train()
        model = AutoModel.from_pretrained(pretrained_mlm)



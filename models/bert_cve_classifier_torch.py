import logging
import math

import daiquiri
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from models.bert_model_weighted import BertForSequenceClassification
from transformers.data.processors import (
    glue_processors as processors,
    glue_output_modes as output_modes,
)

from utils import model_constants as mc

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)


class BertTorchCVEClassifier:
    """Defines the BERT binary classifier using transformers and torch."""

    def __init__(self, trained_model_path: str):
        """Load the transformer model."""
        self.tokenizer = BertTokenizer.from_pretrained(
            trained_model_path, do_lower_case=mc.TOKENIZER_CONVERT_LOWER_CASE
        )
        self.config = BertConfig.from_pretrained(trained_model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            trained_model_path, config=self.config
        )
        self.processor = processors[mc.TASK_NAME]()
        self.output_mode = output_modes[mc.TASK_NAME]

    def predict(
        self, df: pd.DataFrame, batch_size: int = mc.BATCH_SIZE_PROB_CVE_BERT_TORCH
    ) -> np.array:
        """Run inference for the new data."""
        # Turn off batch-norm and dropout layers.
        self.model.eval()
        batch_size = batch_size or mc.BATCH_SIZE_PROB_CVE_BERT_TORCH
        input_tensor = self._preprocess_data(df)
        _logger.debug("Shape of input Tensor: {}".format(input_tensor.shape))
        _logger.info(
            "Running Inference on {} samples with a batch size of {}, num_batches: {}",
            input_tensor.shape[0],
            batch_size,
            math.ceil(input_tensor.shape[0] / batch_size),
        )
        input_dataset = TensorDataset(input_tensor)
        dataloader = DataLoader(input_dataset, batch_size=batch_size)
        preds = None
        for idx, batch in enumerate(dataloader):
            _logger.info("Processing batch #{}".format(idx + 1))
            with torch.no_grad():
                logits = self.model(batch[0])
            if preds is not None:
                preds = torch.cat((preds, logits[0]), dim=0)
            else:
                preds = logits[0]
        preds: np.array = torch.argmax(preds, dim=1).detach().cpu().numpy()
        return preds

    def _preprocess_data(self, df: pd.DataFrame) -> Tensor:
        """Convert the input to format used by the model."""
        # Standard terminology in transformers, an example is any text sample
        features = []
        _logger.info("Converting {} examples to features".format(df.shape[0]))
        for _, description in tqdm(df["norm_description"].iteritems(), total=df.shape[0]):
            # These features are our input_ids for the model.
            features.append(
                self.tokenizer.encode(
                    description, add_special_tokens=True, max_length=512, pad_to_max_length=True,
                )
            )
        _logger.debug("Now converting to a Tensor.")
        feature_tensor = torch.tensor(features, device="cpu")
        _logger.debug("Done.")

        return feature_tensor

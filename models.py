import os
import numpy as np
import sys

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import FocalLossWithLabelSmoothing
import logging

from dataclasses import dataclass
from typing import Any, Dict, List

from .metrics import preprocess_logits_for_metrics as preprocess
from .metrics import compute_cls_metrics, custom_compute_metrics


@dataclass
class DataCollator:
    tokenizer: Any
    padding: bool = True
    max_length: int = None
    dataset_label_field: str = "label"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [example["input_ids"] for example in batch]
        attention_masks = [example["attention_mask"] for example in batch]
        labels = [example[self.dataset_label_field] for example in batch]

        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_encoding["labels"] = torch.tensor(
            [self.tokenizer(
                str(example[self.dataset_label_field]),
                padding=False,
                max_length=2,
                truncation=True,
                return_tensors="pt"
            )["input_ids"][-1][-1] for example in batch],
            dtype=torch.long
        )

        return batch_encoding

class SFTTrainerForSeqCLS(SFTTrainer):
    def __init__(
        self,
        ce_loss_weight=1.0,
        focal_loss_weight=0.0,
        num_classes=1,
        label_balance_logic = False,
        cl_head = True
        dataset_label_field = 'label',
        data_collator = None,
        tokenizer = None,
        preprocess_logits_for_metrics = None,
        compute_metrics = None,
        *args, **kwargs
    ):
        super().__init__(
            tokenizer = tokenizer,
            preprocess_logits_for_metrics =  preprocess_logits_for_metrics if preprocess_logits_for_metrics else preprocess,
            data_collator = data_collator if data_collator else DataCollator(
                tokenizer=tokenizer,
                dataset_label_field=dataset_label_field
            ),
            compute_metrics = compute_metrics if compute_metrics else lambda eval_preds: custom_compute_metrics(eval_preds, tokenizer.pad_token_id),
            *args,
            **kwargs
        )
        self.ce_loss_weight = ce_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.num_classes = torch.tensor(num_classes, dtype=torch.float32)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def focal_loss(self, logits, targets):
        total_instances = self.num_classes.sum()
        class_weights = total_instances / (len(self.num_classes) * self.num_classes)
        class_weights = class_weights / class_weights.sum()  # Normalize
        class_weights = class_weights.to(logits.device)

        loss_fn = FocalLossWithLabelSmoothing(alpha=class_weights, gamma=2.0, smoothing=0.1)
        loss = loss_fn(logits, targets)

        return loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
    
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    
        last_logits = logits[:, -1, :]
        if labels.dim() == 2:
            target_labels = labels[:, -1] 
        else:
            target_labels = labels
        loss = 0
        if self.ce_loss_weight > 0:
            loss += self.ce_loss_weight * F.cross_entropy(last_logits, target_labels)
        if self.focal_loss_weight > 0:
            loss += self.focal_loss_weight * self.focal_loss(last_logits, target_labels)
            
        return (loss, outputs) if return_outputs else loss

    def tokenize_input(self, batch, input_col):
        texts = [item[input_col] for item in batch]
        inputs = self.processing_class(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        return inputs


    def predict(self, test_dataset, batch_size=1, input_col="instruction", top_k=10, **kwargs):
        self.model.eval()
    
        predictions = []
        top_tokens = []
        softmax_scores = []
    
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: self.tokenize_input(batch, input_col=input_col)
        )
    
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
    
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    
                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
    
                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    topk_tokens_batch = [self.processing_class.decode([idx.item()]) for idx in topk_indices[i]]
                    topk_scores_batch = topk_probs[i].tolist()
                
                    predictions.append(topk_tokens_batch[0])
                    softmax_scores.append(topk_scores_batch[0])
                    top_tokens.append(list(zip(topk_tokens_batch, topk_scores_batch)))
    
        return {
            "predictions": predictions,
            "softmax_scores": softmax_scores,
            "top_tokens": top_tokens,
        }



import os
import numpy as np

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import FocalLossWithLabelSmoothing

class SFTTrainerForSeqCLS(SFTTrainer):
    def __init__(
        self,
        ce_loss_weight=1.0,
        focal_loss_weight=0.0,
        num_classes=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ce_loss_weight = ce_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.num_classes = torch.tensor(num_classes, dtype=torch.float32)

    def focal_loss(self, logits, targets):
        total_instances = self.num_classes.sum()
        class_weights = total_instances / (len(self.num_classes) * self.num_classes)
        class_weights = class_weights / class_weights.sum()  # Normalize
        class_weights = class_weights.to(logits.device)

        loss_fn = FocalLossWithLabelSmoothing(alpha=class_weights, gamma=2.0, smoothing=0.1)
        loss = loss_fn(logits, targets)

        return loss

    def custom_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits  #(batch_size, seq_length, vocab_size)
        labels = inputs["labels"]  #(batch_size, seq_length)

        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        
        preds = torch.argmax(logits, dim=-1)  #(batch_size, seq_length)
        print(preds)
        last_preds = []
        last_labels = []
    
        for i in range(preds.shape[0]):
            valid_pred_indices = torch.where(
                (preds[i] != tokenizer.pad_token_id) & (preds[i] != 271) & (preds[i] != 512)
            )[0]
            
            valid_label_indices = torch.where(
                (labels[i] != tokenizer.pad_token_id) & (labels[i] != 271) & (labels[i] != 512)
            )[0]
    
            if len(valid_pred_indices) > 0 and len(valid_label_indices) > 0:
                last_preds.append(preds[i][valid_pred_indices[-1]])
                last_labels.append(labels[i][valid_label_indices[-1]])
    
        if len(last_preds) == 0 or len(last_labels) == 0:
            return torch.tensor(0.0, requires_grad=True)
    
        last_preds = torch.stack(last_preds)
        last_labels = torch.stack(last_labels)

        loss = 0
        if self.ce_loss_weight > 0:
            loss += self.ce_loss_weight * F.cross_entropy(last_preds.unsqueeze(0).float(), last_labels.unsqueeze(0))
        if self.focal_loss_weight > 0:
            loss += self.focal_loss_weight * self.focal_loss(last_preds.unsqueeze(0).float(), last_labels.unsqueeze(0))
            
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

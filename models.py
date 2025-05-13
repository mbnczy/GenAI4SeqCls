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
from .metrics import compute_cls_metrics, custom_compute_metrics, custom_compute_cls_metrics

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

from tqdm.auto import tqdm

from .balancing_methods import LabelBalancedBatchSampler


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
        model,
        labels,
        ce_loss_weight=1.0,
        focal_loss_weight=0.0,
        label_balance_logic = False,
        cl_head = True,
        dataset_label_field = 'label',
        data_collator = None,
        tokenizer = None,
        preprocess_logits_for_metrics = None,
        compute_metrics = None,
        rag_dataset = None,
        model_name_or_path=None,
        rag_model = 'all-MiniLM-L6-v2',
        wandb = None,
        *args, **kwargs
    ):
        self.dataset_label_field = dataset_label_field
        self.label_balance_logic = label_balance_logic
        self.args = args
        self.device = next(model.parameters()).device
        self.cl_head = cl_head
        self.processing_class = tokenizer
        
        try:
            tokenized_labels = [self.processing_class.encode(str(label), add_special_tokens=False)[0] for label in labels]
        except:
            self.processing_class = self.processing_class.tokenizer
            tokenized_labels = [self.processing_class.encode(str(label), add_special_tokens=False)[0] for label in labels]
            
        self.label2tokenid = dict(zip(labels, tokenized_labels))
        self.tokenid2label = dict(zip(tokenized_labels, labels))
        
        if self.cl_head:
            model = self.set_classification_head(model, tokenized_labels)

        if not compute_metrics:
            if self.cl_head:
                compute_metrics = lambda eval_preds: custom_compute_cls_metrics(
                    eval_preds,
                    labels,
                    self.tokenid2label,
                    tokenizer.pad_token_id
                )
            else:
                compute_metrics = lambda eval_preds: custom_compute_metrics(
                    eval_preds,
                    labels,
                    self.processing_class,
                    tokenizer.pad_token_id
                )
        super().__init__(
            model = model,
            tokenizer = tokenizer,
            preprocess_logits_for_metrics =  preprocess_logits_for_metrics if preprocess_logits_for_metrics else preprocess,
            data_collator = data_collator if data_collator else DataCollator(
                tokenizer=tokenizer,
                dataset_label_field=dataset_label_field
            ),
            compute_metrics = compute_metrics,
            *args,
            **kwargs
        )
        self.ce_loss_weight = ce_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.num_classes = torch.tensor(
            tokenized_labels,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device = self.device
        )
        
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"{name}: {param.grad.dtype}")
        
        ## RAG
        self.rag = rag_dataset is not None
        if self.rag:
            self.rag_model = SentenceTransformer(rag_model)
            self.rag_label_to_texts = defaultdict(list)
            self.rag_label_to_faiss = {}
    
            for text, label in zip(rag_dataset['text'], rag_dataset[dataset_label_field]):
                self.rag_label_to_texts[label].append(text)
    
            for label, label_texts in self.rag_label_to_texts.items():
                embeddings = self.rag_model.encode(label_texts, normalize_embeddings=True)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                self.rag_label_to_faiss[label] = (index, label_texts)

    def set_classification_head(self, model, labels):
        #label_tokenids = [self.tokenizer.encode(str(label), add_special_tokens=False)[0] for label in labels]
        try:
            model.lm_head.weight = torch.nn.Parameter(
                torch.vstack(
                    [model.lm_head.weight[tokenid, :].to(torch.float32) for tokenid in labels]
                )
            )
        except:
            model.base_model.model.language_model.lm_head.weight = torch.nn.Parameter(
                torch.vstack(
                    [model.base_model.model.language_model.lm_head.weight[tokenid, :].to(torch.float32) for tokenid in labels]
                )
            )
        #print(model.lm_head.weight.shape)
        return model

    def get_train_dataloader(self):
        if self.label_balance_logic:
            sampler = LabelBalancedBatchSampler(
                labels=self.train_dataset[self.dataset_label_field],
                batch_size=self.args.per_device_train_batch_size
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
            )
        
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
        ## DEBUG
        #print(input_ids)
        #print(logits[~torch.isnan(logits)])
        ##
        last_logits = logits[:, -1, :]
        if labels.dim() == 2:
            target_labels = labels[:, -1] 
        else:
            target_labels = labels

        ## DEBUG
        #print(last_logits)
        #print(target_labels)
        if self.cl_head:
            target_labels = torch.tensor(
                [self.tokenid2label[target_label.item()] for target_label in target_labels],
                device = self.device
            )
        else:
            target_labels = self.processing_class.batch_decode(
                target_labels.unsqueeze(1), skip_special_tokens=True
            )

            target_labels = torch.tensor(
                [self.processing_class.convert_tokens_to_ids(label.strip()) for label in target_labels],
                device=self.device
            )
            
        #print(target_labels)
        ##
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

    def predict(self, test_dataset, batch_size=1, input_col="instruction", top_k=10, rag_weight=0.0, **kwargs):
        self.model.eval()
    
        predictions = []
        top_tokens = []
        softmax_scores = []
    
        model_only_predictions = []
        model_only_scores = []
    
        rag_only_predictions = []
        rag_only_scores = []
    
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: self.tokenize_input(batch, input_col=input_col)
        )
    
        text_loader = DataLoader(test_dataset['text'], batch_size=batch_size)
    
        with torch.no_grad():
            for batch, text_batch in tqdm(zip(dataloader, text_loader), desc="Processing", total=len(dataloader)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
    
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                probs = F.softmax(logits, dim=-1)
    
                if self.cl_head and top_k > len(self.label2tokenid):
                    top_k = len(self.label2tokenid)
    
                batch_size = input_ids.size(0)
    
                for i in range(batch_size):
                    if self.cl_head:
                        model_logits = torch.tensor(
                            #[probs[i, self.label2tokenid[label]].item() for label in self.label2tokenid],
                            [probs[i, j].item() for j in range(len(self.label2tokenid))],
                            device=self.device
                        )
                        model_probs = F.softmax(model_logits, dim=0)
                        labels_list = list(self.label2tokenid.keys())
                    else:
                        model_probs = probs[i]
                        labels_list = [
                            self.processing_class.decode([j], skip_special_tokens=True).strip()
                            for j in range(probs.shape[-1])
                        ]
                        #labels_list = list(range(probs.shape[-1]))
    
                    model_top_val, model_top_idx = torch.max(model_probs, dim=0)

                    model_only_predictions.append(labels_list[model_top_idx.item()])
                    model_only_scores.append(model_top_val.item())
    
                    # ====== RAG similarity-based prediction ======
                    if self.rag and rag_weight!=0.0:
                        rag_input_text = text_batch[i]
                        rag_sim_scores = []
        
                        for label in self.label2tokenid.keys():
                            index, texts = self.rag_label_to_faiss[label]
                            emb = self.rag_model.encode([rag_input_text], normalize_embeddings=True)
                            sims, _ = index.search(emb, k=1)
                            rag_sim_scores.append(sims[0][0])
        
                        rag_sim_tensor = torch.tensor(rag_sim_scores, device=self.device)
                        rag_probs = F.softmax(rag_sim_tensor, dim=0)
        
                        rag_top_val, rag_top_idx = torch.max(rag_probs, dim=0)
                        rag_label_list = list(self.label2tokenid.keys())
                        rag_only_predictions.append(rag_label_list[rag_top_idx.item()])
                        rag_only_scores.append(rag_top_val.item())
        
                        combined_probs = (1 - rag_weight) * model_probs + rag_weight * rag_probs
                    else:
                        rag_only_predictions.append(-1)
                        rag_only_scores.append(-1)
                        combined_probs = model_probs
                    topk_combined, topk_indices_combined = torch.topk(combined_probs, k=top_k)
    
                    final_topk_tokens = [labels_list[j] for j in topk_indices_combined.tolist()]
                    final_topk_scores = topk_combined.tolist()
    
                    predictions.append(final_topk_tokens[0])
                    softmax_scores.append(final_topk_scores[0])
                    top_tokens.append(list(zip(final_topk_tokens, final_topk_scores)))
    
        return {
            "predictions": predictions,
            "softmax_scores": softmax_scores,
            "top_tokens": top_tokens,
            "model_predictions": model_only_predictions,
            "model_scores": model_only_scores,
            "rag_predictions": rag_only_predictions,
            "rag_scores": rag_only_scores,
        }
    
        
        
        
        
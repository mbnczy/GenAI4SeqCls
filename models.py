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


#@dataclass
#class DataCollator:
#    tokenizer: Any
#    label2tokenid: dict
#    padding: bool = True
#    max_length: int = None
#    dataset_label_field: str = "label"
#
#    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#        input_ids = [example["input_ids"] for example in batch]
#        attention_masks = [example["attention_mask"] for example in batch]
#        labels = [example[self.dataset_label_field] for example in batch]
#
#        batch_encoding = self.tokenizer.pad(
#            {"input_ids": input_ids, "attention_mask": attention_masks},
#            padding=self.padding,
#            max_length=self.max_length,
#            return_tensors="pt",
#        )
#
#        #batch_encoding["labels"] = torch.tensor(
#        #    [self.tokenizer(
#        #        str(example[self.dataset_label_field]),
#        #        padding=False,
#        #        max_length=2,
#        #        truncation=True,
#        #        return_tensors="pt"
#        #    )["input_ids"][-1][-1] for example in batch],
#        #    dtype=torch.long
#        #)
#        batch_encoding["labels"] = torch.tensor([self.label2tokenid[int(example[self.dataset_label_field])] for example in batch], dtype=torch.long)
#
#        return batch_encoding
@dataclass
class DataCollator:
    tokenizer: Any
    label2tokenid: dict
    padding: bool = True
    max_length: int = None
    dataset_label_field: str = "label"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract fields in original order
        input_ids = [example["input_ids"] for example in batch]
        attention_masks = [example["attention_mask"] for example in batch]
        labels_raw = [example[self.dataset_label_field] for example in batch]
        labels = torch.tensor(
            [self.label2tokenid[label] for label in labels_raw],
            dtype=torch.long
        )

        # Confirm order is maintained
        #print(f"{labels_raw}->{labels.tolist()}")

        # Pad input sequences
        batch_encoding = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            },
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_encoding["labels"] = labels
        return batch_encoding




class SFTTrainerForSeqCLS(SFTTrainer):
    def __init__(
        self,
        model,
        id2label,
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
        labels = list(id2label.keys())
        self.dataset_label_field = dataset_label_field
        self.label_balance_logic = label_balance_logic
        self.args = args
        self.device = next(model.parameters()).device
        self.cl_head = cl_head
        self.processing_class = tokenizer
        #self.processing_class.add_tokens(sorted(set(str(label) for label in labels)))
        new_tokens = [str(t) for t in labels if str(t) not in set(self.processing_class.get_vocab().keys())]
        if len(new_tokens)>0:
            self.processing_class.add_tokens(new_tokens)
            model.resize_token_embeddings(len(self.processing_class))

        self.id2label = id2label
        try:
            tokenized_labels = [self.processing_class.encode(str(label), add_special_tokens=False)[-1] for label in labels]
        except:
            self.processing_class = self.processing_class.tokenizer
            tokenized_labels = [self.processing_class.encode(str(label), add_special_tokens=False)[-1] for label in labels]
            
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
                    self.processing_class.pad_token_id
                )
            else:
                compute_metrics = lambda eval_preds: custom_compute_metrics(
                    eval_preds,
                    labels,
                    self.processing_class,
                    self.processing_class.pad_token_id
                )
        super().__init__(
            model = model,
            tokenizer = self.processing_class,
            preprocess_logits_for_metrics =  preprocess_logits_for_metrics if preprocess_logits_for_metrics else preprocess,
            data_collator = data_collator if data_collator else DataCollator(
                tokenizer=self.processing_class,
                dataset_label_field=dataset_label_field,
                label2tokenid = self.label2tokenid
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
        ##
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

#    def predict(self, test_dataset, batch_size=1, input_col="instruction", top_k=10, rag_weight=0.0, **kwargs):
#        self.model.eval()
#    
#        predictions = []
#        top_tokens = []
#        softmax_scores = []
#    
#        model_only_predictions = []
#        model_only_scores = []
#    
#        rag_only_predictions = []
#        rag_only_scores = []
#    
#        dataloader = DataLoader(
#            test_dataset,
#            batch_size=batch_size,
#            collate_fn=lambda batch: self.tokenize_input(batch, input_col=input_col)
#        )
#    
#        text_loader = DataLoader(test_dataset['text'], batch_size=batch_size)
#    
#        with torch.no_grad():
#            for batch, text_batch in tqdm(zip(dataloader, text_loader), desc="Processing", total=len(dataloader)):
#                input_ids = batch["input_ids"].to(self.device)
#                attention_mask = batch.get("attention_mask")
#                if attention_mask is not None:
#                    attention_mask = attention_mask.to(self.device)
#    
#                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
#                probs = F.softmax(logits, dim=-1)
#    
#                if self.cl_head and top_k > len(self.label2tokenid):
#                    top_k = len(self.label2tokenid)
#    
#                batch_size = input_ids.size(0)
#    
#                for i in range(batch_size):
#                    if self.cl_head:
#                        model_logits = torch.tensor(
#                            #[probs[i, self.label2tokenid[label]].item() for label in self.label2tokenid],
#                            [probs[i, j].item() for j in range(len(self.label2tokenid))],
#                            device=self.device
#                        )
#                        model_probs = F.softmax(model_logits, dim=0)
#                        labels_list = list(self.label2tokenid.keys())
#                    #else:
#                    #    model_probs = probs[i]
#                    #    labels_list = [
#                    #        self.processing_class.decode([j], skip_special_tokens=True).strip()
#                    #        for j in range(probs.shape[-1])
#                    #    ]
#                    #    #labels_list = list(range(probs.shape[-1]))
#                    else:
#                        label_token_ids = [self.label2tokenid[label] for label in self.label2tokenid]
#                        model_probs = probs[i][label_token_ids]
#                        model_probs = F.softmax(model_probs, dim=0)
#                        labels_list = list(self.label2tokenid.keys())
#
#    
#                    model_top_val, model_top_idx = torch.max(model_probs, dim=0)
#
#                    model_only_predictions.append(labels_list[model_top_idx.item()])
#                    model_only_scores.append(model_top_val.item())
#    
#                    # ====== RAG similarity-based prediction ======
#                    if self.rag and rag_weight!=0.0:
#                        rag_input_text = text_batch[i]
#                        rag_sim_scores = []
#        
#                        for label in self.label2tokenid.keys():
#                            index, texts = self.rag_label_to_faiss[label]
#                            emb = self.rag_model.encode([rag_input_text], normalize_embeddings=True)
#                            sims, _ = index.search(emb, k=1)
#                            rag_sim_scores.append(sims[0][0])
#        
#                        rag_sim_tensor = torch.tensor(rag_sim_scores, device=self.device)
#                        rag_probs = F.softmax(rag_sim_tensor, dim=0)
#        
#                        rag_top_val, rag_top_idx = torch.max(rag_probs, dim=0)
#                        rag_label_list = list(self.label2tokenid.keys())
#                        rag_only_predictions.append(rag_label_list[rag_top_idx.item()])
#                        rag_only_scores.append(rag_top_val.item())
#        
#                        combined_probs = (1 - rag_weight) * model_probs + rag_weight * rag_probs
#                    else:
#                        rag_only_predictions.append(-1)
#                        rag_only_scores.append(-1)
#                        combined_probs = model_probs
#
#                    if top_k > combined_probs.shape[0]:
#                        top_k = combined_probs.shape[0]
#                        
#                    topk_combined, topk_indices_combined = torch.topk(combined_probs, k=top_k)
#    
#                    final_topk_tokens = [labels_list[j] for j in topk_indices_combined.tolist()]
#                    final_topk_scores = topk_combined.tolist()
#    
#                    predictions.append(final_topk_tokens[0])
#                    softmax_scores.append(final_topk_scores[0])
#                    top_tokens.append(list(zip(final_topk_tokens, final_topk_scores)))
#    
#        return {
#            "predictions": predictions,
#            "softmax_scores": softmax_scores,
#            "top_tokens": top_tokens,
#            "model_predictions": model_only_predictions,
#            "model_scores": model_only_scores,
#            "rag_predictions": rag_only_predictions,
#            "rag_scores": rag_only_scores,
#        }
    
    def predict(self, test_dataset, batch_size=1, input_col="instruction", top_k=10, rag_weight=0.0, **kwargs):
        self.model.eval()
        assert not self.model.training
        print(self.label2tokenid)

    
        predictions = []
        softmax_scores = []
        top_tokens = []
    
        model_only_predictions = []
        model_only_scores = []
        model_only_top_tokens = []
    
        rag_only_predictions = []
        rag_only_scores = []
        rag_only_top_tokens = []

        print("test_Dataset order")
        print(test_dataset["label"])
    
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=0
        )
        print("Dataloader order")
        print([batch["labels"] for batch in dataloader])
    
        text_loader = DataLoader(test_dataset['text'], batch_size=batch_size, shuffle=False, num_workers=0)
    
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
                        label_token_ids = [self.label2tokenid[label] for label in self.label2tokenid]
                        model_probs = probs[i][label_token_ids]
                        model_probs = F.softmax(model_probs, dim=0)
                        labels_list = list(self.label2tokenid.keys())
    
                    # ===== Model-only top-k =====
                    model_topk_vals, model_topk_idxs = torch.topk(model_probs, k=min(top_k, model_probs.shape[0]))
                    model_only_predictions.append(labels_list[model_topk_idxs[0].item()])
                    model_only_scores.append(model_topk_vals[0].item())
                    model_only_top_tokens.append(list(zip([labels_list[j] for j in model_topk_idxs.tolist()],model_topk_vals.tolist())))

    
                    # ===== RAG-only prediction (optional) =====
                    if self.rag and rag_weight != 0.0:
                        rag_input_text = text_batch[i]
                        rag_sim_scores = []
    
                        for label in self.label2tokenid.keys():
                            index, _ = self.rag_label_to_faiss[label]
                            emb = self.rag_model.encode([rag_input_text], normalize_embeddings=True)
                            sims, _ = index.search(emb, k=1)
                            rag_sim_scores.append(sims[0][0])
    
                        rag_sim_tensor = torch.tensor(rag_sim_scores, device=self.device)
                        rag_probs = F.softmax(rag_sim_tensor, dim=0)
    
                        rag_topk_vals, rag_topk_idxs = torch.topk(rag_probs, k=min(top_k, rag_probs.shape[0]))
                        rag_labels_list = list(self.label2tokenid.keys())
    
                        rag_only_predictions.append(rag_labels_list[rag_topk_idxs[0].item()])
                        rag_only_scores.append(rag_topk_vals[0].item())
                        rag_only_top_tokens.append(list(zip([rag_labels_list[j] for j in rag_topk_idxs.tolist()],
                                                            rag_topk_vals.tolist())))
    
                        # ===== Combine model & rag =====
                        combined_probs = (1 - rag_weight) * model_probs + rag_weight * rag_probs
                    else:
                        rag_only_predictions.append(-1)
                        rag_only_scores.append(-1)
                        rag_only_top_tokens.append([])
                        combined_probs = model_probs
    
                    # ===== Final combined prediction =====
                    top_k_eff = min(top_k, combined_probs.shape[0])

                    topk_combined, topk_indices_combined = torch.topk(combined_probs, k=top_k_eff)
    
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
            "model_top_tokens": model_only_top_tokens,
    
            "rag_predictions": rag_only_predictions,
            "rag_scores": rag_only_scores,
            "rag_top_tokens": rag_only_top_tokens,
        }
#    def predict(self, test_dataset, batch_size=1, input_col="instruction", pad_token_id=-100, **kwargs):
#        self.model.eval()
#        assert not self.model.training
#    
#        predictions = []
#        decoded_predictions = []
#    
#        dataloader = DataLoader(
#            test_dataset,
#            batch_size=batch_size,
#            collate_fn=self.data_collator,
#            shuffle=False
#        )
#    
#        with torch.no_grad():
#            for batch in tqdm(dataloader, desc="Predicting"):
#                input_ids = batch["input_ids"].to(self.device)
#                attention_mask = batch.get("attention_mask")
#                if attention_mask is not None:
#                    attention_mask = attention_mask.to(self.device)
#    
#                outputs = self.model.generate(
#                    input_ids=input_ids,
#                    attention_mask=attention_mask,
#                    max_new_tokens=10,
#                    do_sample=False
#                )
#
#    
#                for i in range(outputs.shape[0]):
#                    output_seq = outputs[i].cpu().numpy()
#    
#                    # Find last token not equal to pad_token_id
#                    non_pad_indices = np.where(output_seq != pad_token_id)[0]
#                    if len(non_pad_indices) == 0:
#                        last_token_id = pad_token_id
#                    else:
#                        last_token_id = output_seq[non_pad_indices[-1]]
#    
#                    predictions.append(last_token_id)
#    
#                    # Optional: decode to string if needed
#                    decoded = self.processing_class.decode([last_token_id], skip_special_tokens=True)
#                    decoded_predictions.append(decoded)
#    
#        return {
#            #"token_predictions": predictions,
#            "predictions": decoded_predictions,
#        }



        
        
        
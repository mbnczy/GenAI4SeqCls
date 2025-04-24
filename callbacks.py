import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

class LLMSampleCallback(WandbCallback):
    def __init__(self, trainer, test_dataset, batch_size=1, num_samples=10, max_new_tokens=256):
        "A CallBack to log samples as a wandb.Table during training"
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.trainer = trainer
        self.trainer.model = self.trainer.model.cuda()
        self.tokenizer = trainer.processing_class
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens
        )
        self.batch_size = batch_size

    def generate(self, prompt, top_k=5):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to('cuda')

        with torch.inference_mode():
            outputs = self.trainer.model(
                input_ids=tokenized_prompt['input_ids'],
                attention_mask=tokenized_prompt.get("attention_mask")
            )

            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            topk_tokens = [self.tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
            topk_scores = topk_probs[0].tolist()

        for token, score in zip(topk_tokens, topk_scores):
            if '\n' not in token:
                return token, score, list(zip(topk_tokens, topk_scores))

        return topk_tokens[0], topk_scores[0], list(zip(topk_tokens, topk_scores))

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["text", "true_label", "pred_label", "confidence", "top_k_tokens_scores"] + list(self.gen_config.to_dict().keys()))

        for _, example in tqdm(examples.iterrows(), total=len(examples), leave=False):
            #top_token, top_score, top_k_list = self.generate(prompt=prompt)
            
            records_table.add_data(
                example['text'],
                example['y_true'],
                example['y_preds'],
                example['y_probs'],
                str(example['top_tokens']),
                *list(self.gen_config.to_dict().values())
            )

        return records_table

    def log_confusion_matrix(self, y_true, y_preds):
        filtered_true, filtered_preds = zip(*[
            (y, p) for y, p in zip(y_true, y_preds)
            if str(p).isdigit()
        ]) if any(str(p).isdigit() for p in y_preds) else ([], [])

        class_names = sorted(set(filtered_true + filtered_preds), key=int)
    
        for label in filtered_true + filtered_preds:
            assert label in class_names, f"Label {label} not in class_names!"

        
        print(y_true)
        print(y_preds)
        print(filtered_true)
        print(filtered_preds)
        filtered_true = list(map(int, filtered_true))
        filtered_preds = list(map(int, filtered_preds))
        
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=filtered_true,
                preds=filtered_preds,
                #class_names=filtered_preds#class_names
            )
        })

    def on_evaluate(self, args, state, control, **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        
        results = self.trainer.predict(
            self.sample_dataset,
            batch_size=self.batch_size
        )
        examples = self.sample_dataset.to_pandas()
        
        examples['y_true'] = [element['label'] for element in self.sample_dataset]
        examples['y_preds'] = results['predictions']
        examples['y_probs'] = results['softmax_scores']
        examples['top_tokens'] = results['top_tokens']

        self._wandb.log({"sample_predictions": self.samples_table(examples)})

        self.log_confusion_matrix(examples['y_true'].tolist(), examples['y_preds'].tolist())
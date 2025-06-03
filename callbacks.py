import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import time
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Dict, Any
import os
from sklearn.metrics import precision_recall_fscore_support

#class LLMSampleCallback(WandbCallback):
class LLMSampleCallback(TrainerCallback):
    def __init__(
        self,
        #trainer,
        wandb_run,
        test_dataset,
        batch_size=1,
        #num_samples=10,
        #max_new_tokens=256
    ):
        "A CallBack to log samples as a wandb.Table during training"
        super().__init__()
        self.sample_dataset = test_dataset#.select(range(num_samples))
        self.wandb_run = wandb_run
        #self.trainer = trainer
        #self.trainer.model = self.trainer.model.cuda()
        #self.tokenizer = trainer.processing_class
        #self.gen_config = GenerationConfig.from_pretrained(
        #    trainer.model.name_or_path,
        #    max_new_tokens=max_new_tokens
        #)
        #self.num_samples = num_samples
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
            if self.trainer.cl_head:
                print(self.trainer.label2tokenid)
                topk_tokens = [self.tokenizer.decode([self.trainer.label2tokenid[idx.item()]]) for idx in topk_indices[0]]
            else:
                topk_tokens = [self.tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
            topk_scores = topk_probs[0].tolist()

        for token, score in zip(topk_tokens, topk_scores):
            if '\n' not in token:
                return token, score, list(zip(topk_tokens, topk_scores))

        return topk_tokens[0], topk_scores[0], list(zip(topk_tokens, topk_scores))

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(
            columns=["text", "true_label", "pred_label", "confidence", "top_k_tokens_scores", "model_pred","rag_pred","model_score","rag_score","model_top_tokens","rag_top_tokens"]# + list(self.gen_config.to_dict().keys())
        )

        for _, example in tqdm(examples.iterrows(), total=len(examples), leave=False):
            #top_token, top_score, top_k_list = self.generate(prompt=prompt)
            records_table.add_data(
                example['text'],
                example['y_true'],
                example['y_preds'],
                example['y_probs'],
                str(example['top_tokens']),
                example["model_preds"],
                example["rag_preds"],
                example['model_scores'],
                example['rag_scores'],
                str(example['model_top_tokens']),
                str(example['rag_top_tokens']),
                #*list(self.gen_config.to_dict().values())
            )

        return records_table

    def log_confusion_matrix(self, y_true, y_preds, plot_name = "confusion_matrix", label_names=None):
        filtered_true, filtered_preds = zip(*[
            (y, p) for y, p in zip(y_true, y_preds)
            if str(p).isdigit()
        ]) if any(str(p).isdigit() for p in y_preds) else ([], [])

        class_names = sorted(set(filtered_true + filtered_preds), key=int)
    
        for label in filtered_true + filtered_preds:
            assert label in class_names, f"Label {label} not in class_names!"

        filtered_true = list(map(int, filtered_true))
        filtered_preds = list(map(int, filtered_preds))
        
        self.wandb_run.log({
            plot_name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=filtered_true,
                preds=filtered_preds,
                title=plot_name
                #class_names=filtered_preds#class_names
            )
        })
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        #self.trainer = kwargs.get("trainer", None)
        #if self.trainer is None:
        #    print("Trainer not passed in kwargs. Add this callback with trainer.add_callback().")
        #    return
        
        #self.trainer = kwargs.get("trainer")
        #self.trainer.model = self.trainer.model.cuda()
        #self.tokenizer = trainer.processing_class
        #self.gen_config = GenerationConfig.from_pretrained(
        #    trainer.model.name_or_path,
        #    max_new_tokens=self.max_new_tokens
        #)
        
        results = self.trainer.predict(
            self.sample_dataset,
            batch_size=self.batch_size,
            rag_weight=0.5
        )
        examples = self.sample_dataset.to_pandas()
        
        examples['y_true'] = [element['label'] for element in self.sample_dataset]
        examples['y_preds'] = results['predictions']
        examples['y_probs'] = results['softmax_scores']
        examples['top_tokens'] = results['top_tokens']
        examples['model_preds'] = results['model_predictions'] #list(zip(results["model_predictions"], results["model_scores"]))
        examples['rag_preds'] = results['rag_predictions'] #list(zip(results["rag_predictions"], results["rag_scores"]))
        examples['model_scores'] = results['model_scores']
        examples['rag_scores'] = results['rag_scores']
        examples['model_top_tokens'] = results['model_top_tokens']
        examples['rag_top_tokens'] = results['rag_top_tokens']

        #self._wandb.log({"sample_predictions": self.samples_table(examples)})
        self.wandb_run.log({"sample_predictions": self.samples_table(examples)})

        self.log_confusion_matrix(
            [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
            [self.trainer.id2label[int(i)] for i in examples['y_preds'].tolist()]
        )
        self.log_confusion_matrix(
            [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
            [self.trainer.id2label[int(i)] for i in examples['model_preds'].tolist()],
            "model_predictions"
        )
        if self.trainer.rag:
            self.log_confusion_matrix(
                [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
                [self.trainer.id2label[int(i)] for i in examples['rag_preds'].tolist()],
                "rag_predictions"
            )
        

    def log_predictions(self, results, test_dataset=None):
        if test_dataset:
            self.sample_dataset = test_dataset
        examples = self.sample_dataset.to_pandas()
        
        examples['y_true'] = [element['label'] for element in self.sample_dataset]
        examples['y_preds'] = results['predictions']
        examples['y_probs'] = results['softmax_scores']
        examples['top_tokens'] = results['top_tokens']
        examples['model_preds'] = results['model_predictions'] #list(zip(results["model_predictions"], results["model_scores"]))
        examples['rag_preds'] = results['rag_predictions'] #list(zip(results["rag_predictions"], results["rag_scores"]))
        examples['model_scores'] = results['model_scores']
        examples['rag_scores'] = results['rag_scores']
        examples['model_top_tokens'] = results['model_top_tokens']
        examples['rag_top_tokens'] = results['rag_top_tokens']

        self.wandb_run.log({"sample_predictions": self.samples_table(examples)})

        self.log_confusion_matrix(
            [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
            [self.trainer.id2label[int(i)] for i in examples['y_preds'].tolist()]
        )
        self.log_confusion_matrix(
            [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
            [self.trainer.id2label[int(i)] for i in examples['model_preds'].tolist()],
            "model_predictions"
        )
        if self.trainer.rag:
            self.log_confusion_matrix(
                [self.trainer.id2label[int(i)] for i in examples['y_true'].tolist()],
                [self.trainer.id2label[int(i)] for i in examples['rag_preds'].tolist()],
                "rag_predictions"
            )

        try:
            y_true = list(map(int, examples['y_true']))
            y_pred = list(map(int, examples['y_preds']))
    
            precision, recall, f1_scores, support = precision_recall_fscore_support(
                y_true, y_pred, labels=sorted(set(y_true)), zero_division=0
            )
            class_ids = sorted(set(y_true))
    
            data = [[cid, f1] for cid, f1 in zip(class_ids, f1_scores)]
            table = wandb.Table(data=data, columns=["Class ID", "F1 Score"])
    
            self.wandb_run.log({
                "f1_score_per_class": wandb.plot.scatter(
                    table,
                    x="Class ID",
                    y="F1 Score",
                    title="F1 Score per Class"
                )
            })
        except Exception as e:
            print(f"⚠️ Failed to log F1 scatter plot: {e}")


def custom_eta(trainer):
    if len(trainer.state.log_history) > 0:
        last_log = trainer.state.log_history[-1]
        elapsed_time = sum(log.get('train_runtime', 0) for log in trainer.state.log_history)  # Total elapsed time
        percent_complete = (trainer.state.global_step / trainer.state.max_steps) * 100
        
        if percent_complete > 0:
            total_estimated_time = elapsed_time * (100 / percent_complete)
            eta_seconds = total_estimated_time - elapsed_time
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        else:
            eta = "N/A"
    else:
        eta = "N/A"
        
    return eta

class SlackCallback(TrainerCallback):
    def __init__(self, channel_id: str, slack_bot_token: str, wandb_run = None):
        self.ch_id = channel_id
        self.client = WebClient(token=slack_bot_token)
        self.main_thread = None
        self.wandb_run = wandb_run
        
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        message = f"🧬 Training started 🧬\n\n"
        params = {}
        if self.wandb_run:
            params.update({
                "WandB Run": f"<https://wandb.ai/{self.wandb_run.entity}/{self.wandb_run.project}/{self.wandb_run.id}|{self.wandb_run.name}>",
            })
            if "peft_config" in self.wandb_run.config:
                params.update({
                    "Model": self.wandb_run.config["peft_config"]["default"]["base_model_name_or_path"]
                })
        params.update({
            "Epochs": args.num_train_epochs,
            "Train Batch size": f"{args.per_device_train_batch_size} * {args.gradient_accumulation_steps}",
            "Eval Batch size": args.per_device_eval_batch_size,
            "Learning rate": args.learning_rate,
            "Steps": args.max_steps if args.max_steps > 0 else "auto",
        })
        message += "\n".join(f"- {key}: {value}" for key, value in params.items())
        try:
            response = self.client.chat_postMessage(
                channel=self.ch_id,
                text=message,
                mrkdwn=True
            )
            self.main_thread = response['ts']
            print(f"Slack message sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack message: {e.response['error']}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        progress = round((state.global_step / state.max_steps) * 100, 1) if state.max_steps > 0 else "n/a"
        message = (
            f"🧬 Epoch: {round(state.epoch, 1) if state.epoch is not None else '-'}/{args.num_train_epochs if args.num_train_epochs is not None else '-'} \t"
            f"Progress: {progress}% \n\n"
        )
        message += "\n".join(f"- {key}: {value:.4f}" for key, value in metrics.items())

        try:
            reply_response = self.client.chat_postMessage(
                channel=self.ch_id,
                text=message,
                thread_ts=self.main_thread
            )
            print(f"Slack eval update sent: {reply_response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack evaluation message: {e.response['error']}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        try:
            message = "✅ Training finished!"
            self.client.chat_postMessage(
                channel=self.ch_id,
                text=message,
                thread_ts=self.main_thread
            )
            self.client.reactions_add(
                channel=self.ch_id,
                timestamp=self.main_thread,
                name="white_check_mark"
            )
            print("Slack training completion message sent and ✅ reaction added.")
        except SlackApiError as e:
            print(f"Error sending training end Slack message: {e.response['error']}")

    def log_cr(self, cr_path: str):
        """Optional: Call manually to upload classification report file."""
        if not os.path.exists(cr_path):
            print(f"Classification report file not found: {cr_path}")
            return
        try:
            reply_response = self.client.files_upload_v2(
                channel=self.ch_id,
                file=cr_path,
                title="Classification Report",
                thread_ts=self.main_thread
            )
            print(f"Slack file uploaded: {reply_response['file']['name']}")
        except SlackApiError as e:
            print(f"Error uploading classification report: {e.response['error']}")
            
    def log_image(self, image_path: str):
        """Optional: Call manually to upload classification report file."""
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return
        try:
            reply_response = self.client.files_upload_v2(
                channel=self.ch_id,
                file=image_path,
                title=image_path,
                thread_ts=self.main_thread
            )
            print(f"Slack file uploaded: {reply_response['file']['name']}")
        except SlackApiError as e:
            print(f"Error uploading image: {e.response['error']}")

    def on_test(self, model, test_set_name):
        message = f"🧬 Evaluation started 🧬\n\n"
        params = {}
        if self.wandb_run:
            params.update({
                "WandB Run": f"<https://wandb.ai/{self.wandb_run.entity}/{self.wandb_run.project}/{self.wandb_run.id}|{self.wandb_run.name}>",
            })
            if "peft_config" in self.wandb_run.config:
                params.update({
                    "Model": self.wandb_run.config["peft_config"]["default"]["base_model_name_or_path"]
                })
        params.update({
            "Model": model,
            "Eval Set": test_set_name,
        })
        message += "\n".join(f"- {key}: {value}" for key, value in params.items())
        try:
            response = self.client.chat_postMessage(
                channel=self.ch_id,
                text=message,
                mrkdwn=True
            )
            self.main_thread = response['ts']
            print(f"Slack message sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack message: {e.response['error']}")

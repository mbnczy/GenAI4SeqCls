# GenAI4SeqCLS

The __library__ is built as a subclass of Huggingface’s Trainer, tailored for encoder-style fine-tuning of generative transformer models on sequence classification tasks. It enhances the standard training loop with features like a dedicated classification head, RAG (Retrieval-Augmented Generation) integrated predictions, and label-balanced batch sampling. The framework also includes advanced callbacks for improved experiment monitoring and real-time notifications.

## Features

* **Encoder-Style Fine-Tuning**
    Fine-tunes generative transformer models by reframing classification as a token prediction task, where class labels are represented as specific token IDs corresponding to target categories. Rather than generating full token sequences, the model concentrates on the logits of the final token in the output sequence—directly tied to the class label token ID. The loss is calculated solely on these final token logits, allowing precise alignment with the target class. To enhance training effectiveness and handle class imbalance, the framework supports a flexible combination of cross-entropy and focal loss.


* **Classification Head**
    Introduces a dedicated classification head that generates logits exclusively over token IDs representing the target classes. This is achieved by replacing the model’s original `lm_head` weight matrix with a streamlined, smaller matrix containing only the weights corresponding to the label token IDs. By narrowing the output vocabulary to just the class labels, the model becomes both faster and more efficient, producing logits solely over valid class tokens. This elegant refinement aligns the model’s output space directly with the classification task, helping to avoid hallucinations and making the computations more effective and reliable.


* **RAG-Supported Predictions**
    Enables sophisticated Retrieval-Augmented Generation (RAG) workflows, empowering the model to tap into external knowledge sources during inference for enhanced performance. The RAG module transforms input texts into embedding vectors and compares them against precomputed class-specific embeddings using an efficient approximate nearest neighbor search (e.g., FAISS). This yields similarity scores that reflect the relevance of each class to the input, which are then normalized into probability distributions. These retrieval-based relevance signals seamlessly integrate with the model’s own generative logits, enriching the prediction pipeline with auxiliary knowledge and boosting classification accuracy.


* **Label-Balanced Batch Sampler**
  Implements sampling strategies to maintain balanced class representation within each training batch, helping to mitigate class imbalance issues.

* **Extended Callbacks**

  * **WandbCallback**: An enhanced Weights & Biases integration that logs additional metrics, detailed figures, and advanced visualizations.
  * **SlackCallback**: Sends real-time notifications about fine-tuning progress, metrics, logs, and generated figures to Slack channels for seamless monitoring.


## Installation

You can install the framework easily via pip:

```bash
pip install genai4seqcls
```

## Usage

### Preprocessing Utilities

The library provides helpful preprocessing functions to prepare datasets for training and evaluation:

* **`preprocess_instruction`**: Processes raw examples with customizable prompts and tokenization, ensuring inputs fit the model’s expected format.
* **`filter_long_instructions`**: Removes examples where the input length exceeds a specified threshold, helping avoid tokenization issues.
* **`balance`**: Balances the dataset by downsampling labels to achieve uniform sample counts per class, addressing class imbalance.

```python
from datasets import Dataset
from genai4seqcls.preproc import preprocess_instruction, filter_long_instructions, balance

train_dataset = Dataset.from_pandas(data_train).map(
    lambda example: preprocess_instruction(example, prompt, tokenizer, max_seq_length), num_proc=2
)
test_dataset = Dataset.from_pandas(data_test).map(
    lambda example: preprocess_instruction(example, prompt, tokenizer, max_seq_length), num_proc=2
)

train_dataset = filter_long_instructions(train_dataset, threshold=max_seq_length)
test_dataset = filter_long_instructions(test_dataset, threshold=max_seq_length)

train_dataset = balance(train_dataset, label_col='label', n_samples=samples_per_label)
```

---

### Model Loading Example with 🦥 Unsloth 🦥

_This library is fully compatible with **Unsloth**, enabling seamless integration with its advanced model loading and PEFT features._

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "down_proj"
    ],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=max_seq_length,
)
```

### Fine-Tuning Example

```python
from trl import SFTConfig
from genai4seqcls.models import SFTTrainerForSeqCLS

trainer = SFTTrainerForSeqCLS(
    model=model,
    id2label=id2label,
    ce_loss_weight=0.7,
    focal_loss_weight=0.3,
    label_balance_logic=True,
    cl_head=True,
    dataset_label_field='label',
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    rag_dataset=rag_dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=1e-3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=model_dir,
        report_to="wandb",
    ),
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
    ],
)

trainer_stats = trainer.train()

model.save_pretrained(os.path.join(model_dir,"lora"))
tokenizer.save_pretrained(os.path.join(model_dir,"lora"))
```

### Inference Example
```python
from trl import SFTConfig
from genai4seqcls.models import SFTTrainerForSeqCLS

trainer = SFTTrainerForSeqCLS(
    model = model,
    rag_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    id2label = id2label,
    cl_head = False,
    dataset_label_field = 'label',
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    rag_dataset = train_dataset,
    max_seq_length = max_seq_length,
    tokenizer=tokenizer,
    args=SFTConfig(
        report_to="wandb",
        metric_for_best_model="eval_f1",
    ),
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=4),
    ]
)

results = trainer.predict(
    test_dataset,
    batch_size = eval_batch_size,
    rag_weight = 0.5      
)
```
### Callback Examples
```python
import wandb
from trl import SFTConfig
from genai4seqcls.models import SFTTrainerForSeqCLS
from seqcls.callbacks import SlackCallback, LLMSampleCallback

wandb.login(key="...")
wandb_run = wandb.init(
    project="project1",
    name = model_dir
)

slack_callback = SlackCallback( #log metrics in each eval. round in a thread
    channel_id = "...", 
    slack_bot_token = "...",
    wandb_run = wandb_run
)

wandb_callback = LLMSampleCallback( #log a table with samples, model+rag+combined pred, softmax scores, confusion matrices etc.
    wandb_run = wandb_run,
    test_dataset = test_dataset,#balance(test_dataset, label_col='label', n_samples=1),
    batch_size = eval_batch_size
)

trainer = SFTTrainerForSeqCLS(
    model=model,
    id2label=id2label,
    ce_loss_weight=0.7,
    focal_loss_weight=0.3,
    label_balance_logic=True,
    cl_head=True,
    dataset_label_field='label',
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    rag_dataset=rag_dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=1e-3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=model_dir,
        report_to="wandb",
    ),
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=4),
        slack_callback,
        wandb_callback
    ]
)

wandb_callback.set_trainer(trainer)

slack_callback.log_image(os.path.join(model_dir,"stat1.png")) #optional: log a figure in the fine-tuning thread

results = trainer.predict(
    test_dataset,
    batch_size = eval_batch_size,
    rag_weight = 0.5      
)
# in eval. it does the same
wandb_callback.log_predictions(results)
```

## Acknowledgments

Thank you to the Unsloth team for their efficient library for fine-tuning LLMs, and to Hugging Face for their foundational tools.

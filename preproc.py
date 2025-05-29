from datasets import Dataset
def preprocess_text(example, tokenizer, max_seq_length, text_col='text', label_col='label'):
    instruction = example[text_col]
    tokenized = tokenizer(
        instruction,
        truncation=True,
        max_length=max_seq_length
    )
    tokenized["instruction"]=instruction
    tokenized["output"] = tokenizer(
        str(example[label_col]),
        add_special_tokens=False
    )["input_ids"][0]

    return tokenized

def preprocess_instruction(example, prompt_template, tokenizer, max_seq_length, text_col='text', label_col='label'):
    """
    Formats and tokenizes an input example for instruction-based training.

    This function constructs an instruction using the provided prompt template,
    tokenizes the instruction text, and extracts the token ID of the label.
    Returns a dictionary with tokenized input, the original instruction string,
    and the label's token ID.

    Args:
        example (dict): A single data example containing text and label fields.
        prompt_template (str): A string template with placeholders for text and label.
        tokenizer (Tokenizer): The tokenizer to process the inputs.
        max_seq_length (int): Maximum sequence length for tokenization.
        text_col (str): Key in example dict corresponding to input text.
        label_col (str): Key in example dict corresponding to the label.

    Returns:
        dict: A dictionary with tokenized input, the original instruction,
              and the label token ID.
    """
    instruction = prompt_template.format(text=example[text_col], label="")
    tokenized = tokenizer(
        instruction,
        truncation=True,
        max_length=max_seq_length
    )
    tokenized["instruction"]=instruction
    tokenized["output"] = tokenizer(
        str(example[label_col]),
        add_special_tokens=False
    )["input_ids"][0]

    return tokenized

def filter_long_instructions(dataset, threshold=5000):
    """
    Filters out examples from the dataset where the length of 'input_ids' exceeds the threshold.

    Args:
        dataset (Dataset): A Hugging Face Dataset object with 'input_ids' field.
        threshold (int): Maximum allowed length of input_ids.

    Returns:
        Dataset: A filtered dataset containing only examples within the length limit.
    """
    return dataset.filter(lambda example: len(example['input_ids']) <= threshold)

def balance(dataset, label_col='label', n_samples=20):
    """
    Balances the dataset by sampling up to `n_samples` examples per label class.

    This function groups the dataset by label, samples evenly (with a cap),
    shuffles the resulting examples, and returns a new balanced dataset.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.
        label_col (str): Column name to group by (typically the label).
        n_samples (int): Maximum number of samples to retain per label.

    Returns:
        Dataset: A new balanced and shuffled dataset.
    """
    return Dataset.from_pandas((
        dataset.to_pandas().groupby(label_col)
          .apply(lambda x: x.sample(min(len(x), n_samples), random_state=42))
          .reset_index(drop=True)
          .sample(frac=1, random_state=42)
          .reset_index(drop=True)
    ))
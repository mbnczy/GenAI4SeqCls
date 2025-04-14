import torch
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import wandb
from tqdm import tqdm

class LLMSampleCallback(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256):
        "A CallBack to log samples as a wandb.Table during training"
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens
        )

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["instruction"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(example["text"], generation, *list(self.gen_config.to_dict().values()))
        return records_table

    def on_evaluate(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})

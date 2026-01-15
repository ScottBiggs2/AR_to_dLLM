import torch
import transformers
import wandb
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils.sampling import decode_trim

class LogGenerationsCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer, num_samples=4, max_new_tokens=64, steps=32):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.prompts = [
            "The capital of France is",
            "To build a successful startup, you need",
            "The recipe for a chocolate cake includes",
            "In a distant future, robots"
        ]
        self.sampler = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_main_process:
            return
        
        if model is None:
            return

        model.eval()
        if self.sampler is None:
            self.sampler = MDLMSampler(model=model, tokenizer=self.tokenizer)
        else:
            self.sampler.model = model

        device = next(model.parameters()).device
        inputs = [self.tokenizer.encode(p, add_special_tokens=True) for p in self.prompts[:self.num_samples]]
        
        config = MDLMSamplerConfig(
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=0.7
        )
        
        with torch.no_grad():
            outputs = self.sampler.sample(inputs, config=config)
            
        decoded_outputs = decode_trim(self.tokenizer, outputs, inputs)
        
        table = wandb.Table(columns=["Prompt", "Generation"])
        for prompt, gen in zip(self.prompts, decoded_outputs):
            table.add_data(prompt, gen)
        
        wandb.log({"generations": table}, step=state.global_step)
        model.train()

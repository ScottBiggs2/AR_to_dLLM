import torch
import transformers
from accelerate import PartialState
import wandb
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils.sampling import decode_trim
import time
import os
import dllm

logger = dllm.utils.get_default_logger(__name__)

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
        if not PartialState().is_main_process:
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

class TimeoutCheckpointCallback(transformers.TrainerCallback):
    def __init__(self, timeout_hours=7.75): # 7h 45m
        self.timeout_seconds = timeout_hours * 3600
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        if elapsed >= self.timeout_seconds:
            logger.warning(f"Job running for {elapsed/3600:.2f} hours. Forcing checkpoint and stopping.")
            control.should_save = True
            # control.should_training_stop = True # Optional: stop if we want to rely on requeueing 
            # ideally we just save and let it continue until hard kill, 
            # or we stop gracefully. Given "redundancy", let's save. 
        # Actually, we likely want to keep saving or just save once. 
            # Let's bump start_time by 30 mins to allow another save in 30 mins if still running.
            self.start_time = time.time() - self.timeout_seconds + 1800 

class ThroughputCallback(transformers.TrainerCallback):
    """
    Logs steps per second and samples per second at each logging step.
    """
    def __init__(self):
        self.last_log_time = None
        self.last_log_step = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.last_log_time = time.time()
        self.last_log_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 0:
            return
            
        current_time = time.time()
        if self.last_log_time is None:
             self.last_log_time = current_time
             self.last_log_step = state.global_step
             return

        time_diff = current_time - self.last_log_time
        step_diff = state.global_step - self.last_log_step
        
        if time_diff > 0 and step_diff > 0:
            steps_per_sec = step_diff / time_diff
            
            # Global batch size = per_device * world_size * grad_acc
            # args.per_device_train_batch_size is accurate for this process
            # args.world_size is total processes
            # args.gradient_accumulation_steps is configured
            
            global_batch_size = (
                args.per_device_train_batch_size 
                * args.world_size 
                * args.gradient_accumulation_steps
            )
            samples_per_sec = steps_per_sec * global_batch_size
            
            # Update logs dictionary which is sent to wandb
            # We add it to the logs dict so the Trainer picks it up
            if logs is not None:
                logs["throughput/steps_per_sec"] = steps_per_sec
                logs["throughput/samples_per_sec"] = samples_per_sec
                
        self.last_log_time = current_time
        self.last_log_step = state.global_step 

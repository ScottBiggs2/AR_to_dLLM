import torch
import random
from typing import Dict, Any

def apply_unmaskable_prefix(input_ids, labels, mask_token_id, p_mask, prefix_len=None):
    """S1: Never mask tokens in prefix (instruction/prompt)"""
    b, l = input_ids.shape
    device = input_ids.device
    
    # If prefix_len is not provided, we might look for a specific marker or assume first half
    if prefix_len is None:
        prefix_len = l // 2
        
    maskable_mask = labels != -100
    # Create a mask that is only True for positions after prefix_len
    suffix_mask = torch.zeros((b, l), dtype=torch.bool, device=device)
    suffix_mask[:, prefix_len:] = True
    
    maskable_mask = maskable_mask & suffix_mask
    
    masked_mask = (torch.rand((b, l), device=device) < p_mask) & maskable_mask
    noised_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
    
    return noised_input_ids, masked_mask

def apply_truncated_suffix(input_ids, labels, mask_token_id, p_mask, max_truncate_ratio=0.3):
    """S2: Randomly truncate sequence length by masking everything after a point"""
    b, l = input_ids.shape
    device = input_ids.device
    
    maskable_mask = labels != -100
    
    # For each sequence in batch, choose a truncation point
    noised_input_ids = input_ids.clone()
    final_masked_mask = torch.zeros((b, l), dtype=torch.bool, device=device)
    
    for i in range(b):
        truncate_len = int(l * random.uniform(0, max_truncate_ratio))
        if truncate_len > 0:
            trun_start = l - truncate_len
            # Mask everything from trun_start to end
            noised_input_ids[i, trun_start:] = mask_token_id
            final_masked_mask[i, trun_start:] = True
            
    # Also apply the standard random mask to the non-truncated part
    standard_mask = (torch.rand((b, l), device=device) < p_mask) & maskable_mask & (~final_masked_mask)
    noised_input_ids = torch.where(standard_mask, mask_token_id, noised_input_ids)
    final_masked_mask = final_masked_mask | standard_mask
    
    return noised_input_ids, final_masked_mask

def apply_block_masking(input_ids, labels, mask_token_id, p_mask, num_blocks=3, block_size_range=(10, 50)):
    """S3: Mask contiguous spans (realistic infilling)"""
    b, l = input_ids.shape
    device = input_ids.device
    
    maskable_mask = labels != -100
    noised_input_ids = input_ids.clone()
    final_masked_mask = torch.zeros((b, l), dtype=torch.bool, device=device)
    
    for i in range(b):
        for _ in range(num_blocks):
            block_size = random.randint(*block_size_range)
            start_idx = random.randint(0, max(0, l - block_size))
            noised_input_ids[i, start_idx:start_idx + block_size] = mask_token_id
            final_masked_mask[i, start_idx:start_idx + block_size] = True
            
    # Ensure we only mask maskable positions
    final_masked_mask = final_masked_mask & maskable_mask
    noised_input_ids = torch.where(final_masked_mask, mask_token_id, input_ids)
    
    # We can also add some random noise on top if needed, but CoDA emphasizes these blocks
    return noised_input_ids, final_masked_mask

class MaskingCurriculum:
    """Progressive masking curriculum from CoDA"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        
    def get_strategy_probs(self, current_step):
        """Gradually shift from random -> structured masking"""
        progress = min(1.0, current_step / self.total_steps)
        
        # Early: mostly random masking
        # Late: mostly structured (S1, S2, S3) masking
        probs = {
            'random': max(0.1, 1.0 - progress),
            'unmaskable_prefix': min(0.3, progress * 0.4),
            'truncated_suffix': min(0.2, progress * 0.3),
            'block_masking': min(0.4, progress * 0.5)
        }
        # Normalize
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

def apply_curriculum_masking(input_ids, labels, mask_token_id, p_mask, strategy):
    if strategy == 'unmaskable_prefix':
        return apply_unmaskable_prefix(input_ids, labels, mask_token_id, p_mask)
    elif strategy == 'truncated_suffix':
        return apply_truncated_suffix(input_ids, labels, mask_token_id, p_mask)
    elif strategy == 'block_masking':
        return apply_block_masking(input_ids, labels, mask_token_id, p_mask)
    else: # random
        b, l = input_ids.shape
        maskable_mask = labels != -100
        masked_mask = (torch.rand((b, l), device=input_ids.device) < p_mask) & maskable_mask
        noised_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
        return noised_input_ids, masked_mask

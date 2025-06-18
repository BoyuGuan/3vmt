import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
import logging

logger = logging.getLogger(__name__)

class WeightedLossTrainer(Trainer):
    """
    自定义Trainer，支持对videoCaption和subtitleTranslation分别设置loss权重
    """
    
    def __init__(self, 
                 caption_loss_weight: float = 1.0,
                 translation_loss_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.caption_loss_weight = caption_loss_weight
        self.translation_loss_weight = translation_loss_weight
        self.separator_token = "<translation>"
        
        # 在tokenizer中获取分隔符的token id
        if hasattr(self.tokenizer, 'encode'):
            # 编码分隔符，去掉特殊token
            separator_tokens = self.tokenizer.encode(self.separator_token, add_special_tokens=False)
            if separator_tokens:
                self.separator_token_id = separator_tokens[0]
            else:
                logger.warning(f"Cannot find token id for separator '{self.separator_token}', using fallback method")
                self.separator_token_id = None
        else:
            self.separator_token_id = None
            
        logger.info(f"WeightedLossTrainer initialized with caption_weight={caption_loss_weight}, "
                   f"translation_weight={translation_loss_weight}")
        if self.separator_token_id is not None:
            logger.info(f"Separator token '{self.separator_token}' has id: {self.separator_token_id}")

    def find_separator_positions(self, labels: torch.Tensor) -> torch.Tensor:
        """
        找到labels中<translation>分隔符的位置
        
        Args:
            labels: shape (batch_size, seq_len)
            
        Returns:
            positions: shape (batch_size,) 每个样本中分隔符的位置，-1表示未找到
        """
        batch_size, seq_len = labels.shape
        positions = torch.full((batch_size,), -1, dtype=torch.long, device=labels.device)
        
        if self.separator_token_id is not None:
            # 直接查找token id
            for i in range(batch_size):
                mask = (labels[i] == self.separator_token_id)
                if mask.any():
                    positions[i] = mask.nonzero(as_tuple=False)[0].item()
        else:
            # 回退方法：将labels转回文本查找
            for i in range(batch_size):
                try:
                    # 只考虑非IGNORE_INDEX的部分
                    valid_mask = labels[i] != -100
                    if valid_mask.any():
                        valid_tokens = labels[i][valid_mask]
                        decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
                        if self.separator_token in decoded_text:
                            # 重新编码找到确切位置
                            tokens = self.tokenizer.encode(decoded_text, add_special_tokens=False)
                            separator_tokens = self.tokenizer.encode(self.separator_token, add_special_tokens=False)
                            if separator_tokens:
                                for j in range(len(tokens) - len(separator_tokens) + 1):
                                    if tokens[j:j+len(separator_tokens)] == separator_tokens:
                                        # 映射回原始序列位置
                                        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
                                        if j < len(valid_indices):
                                            positions[i] = valid_indices[j].item()
                                        break
                except Exception as e:
                    logger.warning(f"Error finding separator in sample {i}: {e}")
                    continue
                    
        return positions

    def compute_weighted_loss(self, 
                            logits: torch.Tensor, 
                            labels: torch.Tensor,
                            separator_positions: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失
        
        Args:
            logits: shape (batch_size, seq_len, vocab_size)
            labels: shape (batch_size, seq_len)
            separator_positions: shape (batch_size,) 分隔符位置
            
        Returns:
            weighted_loss: 加权后的总损失
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 将logits和labels展平用于计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算每个位置的损失，不进行reduction
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        token_losses = loss_fct(flat_logits, flat_labels)  # shape: (batch_size * (seq_len-1))
        
        # reshape回原始形状
        token_losses = token_losses.view(batch_size, seq_len - 1)
        
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            sep_pos = separator_positions[i].item()
            sample_losses = token_losses[i]
            valid_mask = (shift_labels[i] != -100)
            
            if not valid_mask.any():
                continue
                
            if sep_pos > 0 and sep_pos < seq_len - 1:  # 找到了分隔符
                # caption部分：开始到分隔符
                caption_mask = valid_mask & (torch.arange(seq_len - 1, device=labels.device) < sep_pos)
                # translation部分：分隔符后到结束
                translation_mask = valid_mask & (torch.arange(seq_len - 1, device=labels.device) >= sep_pos)
                
                # 计算各部分损失
                if caption_mask.any():
                    caption_loss = sample_losses[caption_mask].mean()
                    total_loss += self.caption_loss_weight * caption_loss
                
                if translation_mask.any():
                    translation_loss = sample_losses[translation_mask].mean()
                    total_loss += self.translation_loss_weight * translation_loss
                    
            else:
                # 没有找到分隔符，使用默认权重（可能是纯翻译任务）
                if valid_mask.any():
                    sample_loss = sample_losses[valid_mask].mean()
                    total_loss += sample_loss
                    
            valid_samples += 1
            
        if valid_samples > 0:
            total_loss = total_loss / valid_samples
        else:
            total_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
            
        return total_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法以支持加权损失计算
        """
        labels = inputs.get("labels")
        
        # 前向传播
        outputs = model(**inputs)
        
        if labels is not None:
            logits = outputs.get("logits")
            
            # 找到分隔符位置
            separator_positions = self.find_separator_positions(labels)
            
            # 计算加权损失
            loss = self.compute_weighted_loss(logits, labels, separator_positions)
            
            # 将损失放回outputs中
            outputs.loss = loss
        else:
            # 如果没有labels，使用默认损失
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
            
        return (loss, outputs) if return_outputs else loss 
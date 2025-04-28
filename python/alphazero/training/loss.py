"""
Loss functions for training AlphaZero neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroLoss(nn.Module):
    """
    Combined loss function for AlphaZero training.
    
    Combines policy loss (cross-entropy), value loss (MSE), and 
    L2 regularization to train both policy and value heads.
    
    Args:
        l2_reg (float): L2 regularization coefficient. Default: 1e-4
    """
    
    def __init__(self, l2_reg=1e-4):
        super(AlphaZeroLoss, self).__init__()
        self.l2_reg = l2_reg
    
    def forward(self, policy_logits, value_output, policy_target, value_target, model=None):
        """
        Calculate the combined loss.
        
        Args:
            policy_logits (torch.Tensor): Policy head output (before softmax)
            value_output (torch.Tensor): Value head output
            policy_target (torch.Tensor): Target policy distribution
            value_target (torch.Tensor): Target value (-1 to 1)
            model (nn.Module, optional): Model for L2 regularization
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss, l2_loss)
        """
        # Policy loss (cross-entropy)
        # We use KL divergence with target distribution since we have full distribution,
        # not just a single class label
        policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_output.squeeze(-1), value_target)
        
        # L2 regularization
        l2_loss = torch.tensor(0.0, device=policy_logits.device, requires_grad=True)
        if model is not None and self.l2_reg > 0:
            for param in model.parameters():
                l2_loss = l2_loss + torch.norm(param)**2
            l2_loss = self.l2_reg * l2_loss
        
        # Combined loss
        total_loss = policy_loss + value_loss + l2_loss
        
        return total_loss, policy_loss, value_loss, l2_loss
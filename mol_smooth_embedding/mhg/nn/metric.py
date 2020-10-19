import os
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable, Function
from torch.nn import functional as F


#calculate Log ratio loss
def LogRatio_loss(tensor, label):
    
    log_ratio_loss_sum = torch.tensor(0.0)  
    epsilon = 1e-6
    eps = 1.0
    
    for ida, anchor in enumerate(tensor):
        
        anchor_label_diff = torch.zeros(64)
        anchor_tensor_diff = torch.zeros(64)
        
        for id in range(64):
            
            anchor_label_distance = torch.abs(label[id] - label[ida])
            anchor_label_distance = torch.pow(anchor_label_distance, 2).sum()
            anchor_label_diff[id] = torch.pow(anchor_label_distance + epsilon, 1. /2)

            anchor_tensor_distance = torch.abs(tensor[id] - anchor)
            anchor_tensor_distance = torch.pow(anchor_tensor_distance, 2).sum()
            anchor_tensor_diff[id] = torch.pow(anchor_tensor_distance + epsilon, 1. /2)
        
        sorted_anchor_tensor_diff, sorted_tensor_idx = torch.sort(anchor_tensor_diff)
        semi_sorted_anchor_tensor_diff = sorted_anchor_tensor_diff[0:16]
        semi_sorted_tensor_idx = sorted_tensor_idx[0:16]
        semi_anchor_label_diff = anchor_label_diff[semi_sorted_tensor_idx]
 
        semi_sorted_anchor_label_diff, semi_sorted_label_idx = torch.sort(semi_anchor_label_diff)
    
        for pos_id, pos_label_diff in zip(semi_sorted_tensor_idx[semi_sorted_label_idx[0:16:2]], semi_sorted_anchor_label_diff[0:16:2]):
            
            if ida != pos_id:
            
                L2_tensor_ap = torch.log(anchor_tensor_diff[pos_id] + eps)
                L2_label_ap = torch.log(pos_label_diff + eps)
            
                for idx, (neg_id, neg_label_diff) in enumerate(zip(semi_sorted_tensor_idx[semi_sorted_label_idx[-16:]], semi_sorted_anchor_label_diff[-16:])):
                    
                    if (idx + 1) % 2 == 0:
                
                        L2_tensor_an = torch.log(anchor_tensor_diff[neg_id] + eps)
                        L2_label_an = torch.log(neg_label_diff + eps)
                
                        diff_log_tesor_dist = L2_tensor_ap - L2_tensor_an
                        diff_log_label_dist = L2_label_ap - L2_label_an
                    
                        log_ratio_loss = (diff_log_tesor_dist - diff_log_label_dist).pow(2)
                
                        log_ratio_loss_sum = log_ratio_loss_sum.to("cuda:0") + log_ratio_loss.to("cuda:0")
                    

    return log_ratio_loss_sum


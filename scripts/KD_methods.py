import torch.nn as nn
import torch.nn.functional as F

# Re-implementaion of the vanilla KD loss function
class KD_Hinton(nn.Module):
    def __init__(self, T):
        super().__init__()
        # T: distillation temperture
        self.T = T

    def forward(self, f_s, f_t):
        # f_s: student's output logits
        # f_t: teacher's output logits
        
        p_s = F.log_softmax(f_s/self.T, 1)
        p_t = F.softmax(f_t/self.T, 1)

        # compute the K-L divergence between for those two distrubution
        KD_loss = F.kl_div(p_s, p_t, reduction='batchmean')
        
        return (self.T**2)*KD_loss # scale by T^2


# slightly modify DIST (https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py)
# removed softmax (better results when operating directly on the logits)
# This is used for the SRD loss design in method 'SRDwithDIST', replacing MSE()

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps) 

def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()

def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1)) 

class DISTV2(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0):
        super(DISTV2, self).__init__()
        self.beta = beta    # inter-class loss weight
        self.gamma = gamma  # intra-class loss weight

    def forward(self, z_s, z_t):
        
        # CHANGE: Do not do softmax, apply directly to logits 
        inter_loss = inter_class_relation(z_s, z_t)
        intra_loss = intra_class_relation(z_s, z_t)
        
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss

        return kd_loss


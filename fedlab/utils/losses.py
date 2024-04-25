import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float64, device="cuda")
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.weight = weight

    def forward(self, x, target):
        device = x.device
        print(device)
        x_m = x + self.m_list.to(device=device)
        return F.cross_entropy(x_m, target, weight=self.weight)


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_KD, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float64, device="cuda")
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x, target, soft_target, w_kd):
        device = x.device
        print(device)
        x_m = x + self.m_list.to(device=device)
        log_pred = torch.log_softmax(x_m, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, 0), log_pred)

        kl = F.kl_div(log_pred, soft_target, reduction='batchmean')

        return w_kd * kl + (1 - w_kd) * F.nll_loss(log_pred, target)

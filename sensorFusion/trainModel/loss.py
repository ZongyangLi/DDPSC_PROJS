import torch
import torch.nn as nn
from torch.nn.modules.loss import TripletMarginLoss as OriginTripletMarginLoss
from torch.nn.modules.loss import _Loss
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning import miners

class TripletMarginLoss(OriginTripletMarginLoss):
    def forward(self, input, target):
        # get the input from args
        return super().forward(*input)
    
class MyNCALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_scale = 3.5
        
    def forward(self, input, target):
        
        # noamalize embeddings
        embeddings = [torch.nn.functional.normalize(d, p=2, dim=1) for d in input]
        
        return self.nca_computation(embeddings)
    
    def nca_computation(self, embeddings):
        
        batch_size = list(embeddings[0].size())[0]
        
        loss = 0
        
        for i in range(batch_size):
            anchor_embedding = embeddings[0][i]
            positive_embedding = embeddings[1][i]
            negative_embedding = embeddings[2][i]
            # dot product
            s_ap = torch.matmul(anchor_embedding, positive_embedding)
            s_an = torch.matmul(anchor_embedding, negative_embedding)
            # NCA loss
            exp_sap = torch.exp(s_ap*self.softmax_scale)
            exp_san = torch.exp(s_an*self.softmax_scale)
            loss +=  -torch.log(exp_sap/(exp_sap+exp_san))
        
        loss = loss / float(batch_size)
        
        return loss
    
class HardNegTripletMarginLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.miner = miners.BatchHardMiner()
        self.loss = pml_losses.TripletMarginLoss()
        
    def forward(self, embeddings, labels):
        labels = labels['plot_int']
        miner_output = self.miner(embeddings, labels)
        return self.loss(embeddings, labels, miner_output)


class CenterOfMessPosetLoss(_Loss):
    def __init__(self, margin=0.5, softmax_input=True, size_average=None, reduce=None, reduction='mean'):
        super(CenterOfMessPosetLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        if softmax_input:
            self.softmax = nn.Softmax(1)
        else:
            self.softmax = None
        pass

    def forward(self, input, target):
        small = input[0]
        large = input[1]
        if self.softmax is not None:
            small = self.softmax(small)
            large = self.softmax(large)
        assert len(small.shape) == 2 and len(large.shape) == 2, ValueError('input dimension must be B*C')
        channel_num = torch.arange(1, small.shape[-1] + 1, device=small.device).T
        small_center_of_mass = (small * channel_num).mean(1)
        large_center_of_mass = (large * channel_num).mean(1)
        loss = torch.clamp_min(self.margin + small_center_of_mass - large_center_of_mass, min=0)
        if self.reduction == 'mean':
            return  loss.mean()
        elif self.reduction == 'sum':
            return  loss.sum()

class PosetLoss(_Loss):
    def __init__(self,margin, size_average=None, reduce=None, reduction='mean'):
        super(PosetLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        pass

    def forward(self, input, target):
        small = input[0]
        large = input[1]
        assert len(small.shape) == 2 and len(large.shape) == 2, ValueError('input dimension must be B*C')
        diff = torch.clamp_min(small - large + self.margin, min=0)
        loss = diff.sum(1)/small.shape[1]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        

class HardNegNCALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.miner = miners.BatchHardMiner()
        self.loss = NCALossBase(softmax_scale=10)
        
    def forward(self, embeddings, labels):
        labels = labels['plot_int']
        miner_output = self.miner(embeddings, labels)
        return self.loss(embeddings, labels, miner_output)
    
class NCALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = NCALossBase(softmax_scale=10)
        
    def forward(self, embeddings, labels):
        labels = labels['plot_int']
        return self.loss(embeddings, labels)

class NCALossBase(pml_losses.BaseMetricLossFunction):

    def __init__(self, softmax_scale=1, **kwargs):

        super().__init__(**kwargs)

        self.softmax_scale = softmax_scale
        
    def forward(self, embeddings, labels, indices_tuple=None):
        
        return super().forward(embeddings, labels, indices_tuple)


    def compute_loss(self, embeddings, labels, indices_tuple):

        if len(embeddings) <= 1:

            return self.zero_losses()

        return self.nca_computation(embeddings, embeddings, labels, labels, indices_tuple)

    def nca_computation(self, query, reference, query_labels, reference_labels, indices_tuple):

        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels)

        x = lmu.sim_mat(query, reference)

        if query is reference:

            diag_idx = torch.arange(query.size(0))

            x[diag_idx, diag_idx] = float('-1')

        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).float()

        exp = torch.nn.functional.softmax(self.softmax_scale*x, dim=1)

        exp = torch.sum(exp * same_labels, dim=1)

        non_zero = exp!=0

        loss = torch.mean(-torch.log(exp[non_zero])*miner_weights[non_zero])

        return loss

'''
    def nca_computation(self, query, reference, query_labels, reference_labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels)
        x = -lmu.dist_mat(query, reference, squared=True)
        if query is reference:
            diag_idx = torch.arange(query.size(0))
            x[diag_idx, diag_idx] = float('-inf')
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).float()
        exp = torch.nn.functional.softmax(self.softmax_scale*x, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp!=0
        loss = torch.mean(-torch.log(exp[non_zero])*miner_weights[non_zero])
        return loss
'''

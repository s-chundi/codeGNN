import torch
import torch.nn as nn
class PairwiseCircleLoss(nn.Module):
    def __init__(self, gamma, margin) -> None:
        super().__init__()
        self.gamma = gamma
        self.margin = margin

    def dot_sim(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        return torch.mm(x, y.t())
    
    def get_classes(self, x:torch.Tensor , y:torch.Tensor):
        similarity_matrix = x @ x.T
        label_matrix = torch.eq(y.view(-1, 1), y.view(1, -1))

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = ~(label_matrix).triu(diagonal=1)

        #flatten
        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        sp = similarity_matrix[positive_matrix] 
        sn = similarity_matrix[negative_matrix]
        return sp,sn
    """
    p: positive examples
    n: negative examples
    q: query
    """
    def forward(self, p: torch.Tensor ,n: torch.Tensor , q: torch.Tensor):
        sp = self.dot_sim(q,p) 
        sn = self.dot_sim(q,n)

        alpha_p = torch.relu(- sp.detach() + 1 + self.margin)
        alpha_n = torch.relu(sn.detach() + self.margin)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = torch.reshape(self.gamma * alpha_p * (sp - delta_p), (-1, 1))
        logit_n = torch.reshape(self.gamma * alpha_n * (sp - delta_n), (-1, 1))

        # label_p = torch.ones_like(logit_p)
        # label_n = torch.zeros_like(logit_n)

        loss = torch.log( 1 + torch.logsumexp(logit_n, -1) + torch.logsumexp(logit_p, -1))
        return loss









    
class CircleLoss(nn.Module):
    def __init__(self, gamma, margin) -> None:
        super().__init__()
        self.gamma = gamma
        self.margin = margin


    def forward(self, x:torch.Tensor , y:torch.Tensor): #x is normalized already
        #pairwise similarity: s_n(j) = x_n(j)^T x / (||x_n(j)|| ||x||)
        similarity_matrix = x @ x.T
        label_matrix = torch.eq(y.view(-1, 1), y.view(1, -1))

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = ~(label_matrix).triu(diagonal=1)

        #flatten
        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        sp = similarity_matrix[positive_matrix] 
        sn = similarity_matrix[negative_matrix]
   
        alpha_p = torch.relu(- sp.detach() + 1 + self.margin)
        alpha_n = torch.relu(sn.detach() + self.margin)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - alpha_p * (sp - delta_p) * self.gamma
        logit_n = alpha_n * (sn - delta_n) * self.gamma

        loss = torch.log( 1 + torch.logsumexp(logit_n, -1) + torch.logsumexp(logit_p, -1))
        print(loss)
        return loss
        
        

#s_n (L) between class, s_p (K) is in class

if __name__ == "__main__":
    x = nn.functional.normalize(torch.rand(256, 64, requires_grad=True)) ## features
    y = torch.randint(high=10, size=(256,)) ## labels - class

    ## hyper paramters gamma [32 -> 1024] m, [-.2 -> .3]

    """
    hyper paramters gamma [32 -> 1024] m, [-.2 -> .3]
    In circle loss paper, they use K (inclass) = 5, P (between class) = 16
    """

    cl = CircleLoss(gamma= 256, margin=0.25)
    cl(x,y)


    
    pcl = PairwiseCircleLoss(gamma= 256, margin=0.25)


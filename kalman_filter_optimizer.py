import torch
from torch.optim.optimizer import Optimizer

class KalmanFilterOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, q=1e-5, r=1e-3):
        defaults = dict(lr=lr)
        super(KalmanFilterOptimizer, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['x'] = torch.zeros_like(p.data)
                state['P'] = torch.ones_like(p.data)
                state['q'] = torch.full_like(p.data, q)
                state['r'] = torch.full_like(p.data, r)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                x = state['x']
                P = state['P']
                q = state['q']
                r = state['r']
                
                # Predict
                x_pred = x
                P_pred = P + q
                
                # Update
                K = P_pred / (P_pred + r)
                x_new = x_pred + K * (grad - x_pred)
                P_new = (1 - K) * P_pred
                
                state['x'] = x_new
                state['P'] = P_new
                
                # Update parameters
                p.data -= lr * x_new
        
        return loss

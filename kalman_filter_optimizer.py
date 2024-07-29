import torch
from torch.optim.optimizer import Optimizer

class KalmanFilterOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, A=None, H=None, Q=None, R=None, q=1e-5, r=1e-3):
        defaults = dict(lr=lr)
        super(KalmanFilterOptimizer, self).__init__(params, defaults)
        
        self.A = A if A is not None else torch.eye(len(params))
        self.H = H if H is not None else torch.eye(len(params))
        self.Q = Q if Q is not None else torch.eye(len(params)) * 1e-5
        self.R = R if R is not None else torch.eye(len(params)) * 1e-3
        
        self.state['x'] = torch.zeros(len(params), 1)
        self.state['P'] = torch.eye(len(params))

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
                
                # Predict
                x_pred = self.A @ self.state['x']
                P_pred = self.A @ self.state['P'] @ self.A.T + self.Q
                
                # Update
                y = grad.view(-1, 1)
                K = P_pred @ self.H.T @ torch.inverse(self.H @ P_pred @ self.H.T + self.R)
                self.state['x'] = x_pred + K @ (y - self.H @ x_pred)
                self.state['P'] = (torch.eye(len(p)) - K @ self.H) @ P_pred
                
                # Update parameters
                p.data -= lr * self.state['x'].view_as(p.data)
        
        return loss

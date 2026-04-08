###This code is almost the same as that in ppo_nonconvex, 
###the difference is that this code uses TruncatedNormal, 
###whereas ppo_nonconvex uses normal

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchrl.modules import TruncatedNormal

# Policy network: Gaussian over 2D vector (x, y)
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.log_std = nn.Parameter(torch.tensor([0.0, 0.0]))  # elementwise std

    def forward(self):
        std = torch.exp(self.log_std)
        dist = TruncatedNormal(self.mean, std,low=torch.tensor([-1, -1]),
                                                high=torch.tensor([2, 1]))
        return dist

#Reward function: negative of the total objective
def reward_fn(samples):
    x = samples[:, 0]
    y = samples[:, 1]
    z = 1.0*np.cos(x)*np.sin(y)-x/(1+y**2)
    return -z

# PPO hyperparameters
clip_eps = 0.2
epochs = 20000
batch_size = 5000
ppo_steps = 10
lr = 0.001
# Initialize policy and optimizer
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=lr)

for epoch in range(epochs):
    # policy to sample 
    dist = policy()
    samples = dist.sample((batch_size,))  # shape: (batch_size, 2)
    old_log_probs = dist.log_prob(samples).detach()  # joint log-prob
    
    rewards = reward_fn(samples)

    for _ in range(ppo_steps):
        dist_new = policy()
        # new_log_probs = dist_new.log_prob(samples).sum(dim=1)  # joint log-prob
        new_log_probs = dist_new.log_prob(samples)  # joint log-prob
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO clipped surrogate objective
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * rewards
        loss = -torch.min(ratio * rewards, clipped).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        mu = policy.mean.detach().numpy()
        print(f"Epoch {epoch:02d} | Mean x: {mu[0]:.4f} | Mean y: {mu[1]:.4f}")

# Final solution
final_xy = policy.mean.detach().numpy()
print(f"\nOptimized x ≈ {final_xy[0]:.4f}, y ≈ {final_xy[1]:.4f}")

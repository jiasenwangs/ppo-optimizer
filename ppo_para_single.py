import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

### Simple PPO code for a single-decision-variable parametrized optimization problem

# Policy network: input is a, output is mean and std of x (decision variable)
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.tensor([0.0]))  # shared log std for stability

    def forward(self, a_batch):
        h = self.net(a_batch)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean.squeeze(-1), std)

# Reward function
def reward_fn(x, a):
    return -a * (x - 2) ** 2 * (a>=0.5) -a * (x - 4) ** 2 * (a<0.5)

# PPO hyperparameters
clip_eps = 0.2
epochs = 10000
batch_size = 100
ppo_steps = 10
lr = 0.0001

# Initialize policy and optimizer
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=lr)

for epoch in range(epochs):
    # Sample input a (parameters, not decision variable) uniformly from [-1, 1]
    a_batch = torch.FloatTensor(batch_size, 1).uniform_(0.1, 1)
    
    dist = policy(a_batch)
    x_samples = dist.sample().clamp(-5, 5)  # clip x to [-5, 5]
    old_log_probs = dist.log_prob(x_samples).detach()
    # Compute rewards
    rewards = reward_fn(x_samples, a_batch.squeeze(-1))
    
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    for _ in range(ppo_steps):
        dist_new = policy(a_batch)
        new_log_probs = dist_new.log_prob(x_samples)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * rewards
        loss = -torch.min(ratio * rewards, clipped).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        # test_a = torch.tensor([[1.0], [-1.0], [0.0]])
        test_a = torch.tensor([[1.0], [0.8], [0.1]])
        test_dist = policy(test_a)
        print(f"Epoch {epoch:03d}")
        for i, a_val in enumerate(test_a):
            mean = test_dist.mean[i].item()
            print(f"  a = {a_val.item():>5.2f} | predicted x ≈ {mean:>6.3f}")

# Try a range of a values
print("\nFinal learned policy predictions:")
for a_val in np.linspace(-1, 1, 5):
    a_tensor = torch.tensor([[a_val]], dtype=torch.float32)
    dist = policy(a_tensor)
    print(f"a = {a_val:>5.2f} → x ≈ {dist.mean.item():.3f}")  #Optimal solution: if a>0.5, x=2; if a<0.5, x=4

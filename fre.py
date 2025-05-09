import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import reward_functions


# —— 你的模型定义（同前） —— 
class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 state_dim: int,
                 num_discrete_embeddings: int,
                 decoder_output_dim: int,
                 emb_dim: int = 128,
                 transformer_layers: int = 2,
                 transformer_heads: int = 2,
                 transformer_mlp_dim: int = 256,
                 hidden_dim: int = 512,
                 reward_min: float = 0.0,
                 reward_max: float = 1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_discrete_embeddings = num_discrete_embeddings
        self.reward_min = reward_min
        self.reward_max = reward_max

        self.state_embed = nn.Linear(state_dim, emb_dim // 2)
        self.reward_embed = nn.Embedding(num_discrete_embeddings, emb_dim // 2)
        self.input_proj = nn.Linear(emb_dim, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            batch_first=True,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=transformer_layers)

        self.fc_mean    = nn.Linear(emb_dim, emb_dim)
        self.fc_log_std = nn.Linear(emb_dim, emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, decoder_output_dim)
        )

    def get_transformer_encoding(self, reward_state_pairs: torch.Tensor):
        B, N, _ = reward_state_pairs.shape
        reward_states = reward_state_pairs[:, :, :-1]  # [B,N,state_dim]
        reward_values = reward_state_pairs[:, :, -1]   # [B,N]

        # 归一化到 [-1,1]
        denom = self.reward_max - self.reward_min
        if denom == 0:
            rv_norm = torch.zeros_like(reward_values)
        else:
            rv_norm = (reward_values - self.reward_min)/denom*2 -1
            rv_norm = torch.clamp(rv_norm, -1.0, 1.0)

        # 离散化索引
        vals_np = rv_norm.detach().cpu().numpy()
        idx_np = np.floor((vals_np/2.0+0.5)*self.num_discrete_embeddings).astype(np.int64)
        idx_np = np.clip(idx_np, 0, self.num_discrete_embeddings-1)
        idx = torch.from_numpy(idx_np).to(reward_state_pairs.device)

        # Embedding
        state_emb = self.state_embed(reward_states)  # [B,N,emb_dim//2]
        val_emb   = self.reward_embed(idx)           # [B,N,emb_dim//2]
        x = torch.cat([state_emb, val_emb], dim=-1)  # [B,N,emb_dim]
        x = self.input_proj(x)

        # Transformer expects [seq_len, B, emb_dim]
        x = x.permute(1,0,2)
        h = self.transformer(x)
        h = h.permute(1,0,2)  # [B,N,emb_dim]

        pooled = h.mean(dim=1)    # [B,emb_dim]
        w_mean    = self.fc_mean(pooled)
        w_log_std = self.fc_log_std(pooled)
        return w_mean, w_log_std

    def forward(self, reward_state_pairs: torch.Tensor):
        w_mean, w_log_std = self.get_transformer_encoding(reward_state_pairs)
        z = w_mean
        y = self.decoder(z)
        return y, (w_mean, w_log_std)

# —— 一个简单的 Dataset 示例 —— 
class RewardPairDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        data: np.ndarray, shape [num_samples, N, state_dim+1]
        """
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # 作为 demo，输入和解码都用同一个 data
        sample = self.data[i]
        return torch.from_numpy(sample), torch.from_numpy(sample)

# —— 损失计算函数 —— 
def compute_vae_loss(model, batch, kl_weight):
    reward_state_pairs, reward_pairs_decode = batch
    reward_pairs_decode = reward_pairs_decode.to('cuda')
    reward_pred, (w_mean, w_log_std) = model(reward_state_pairs)
    # 预测的 reward
    reward_truths = reward_pairs_decode[..., -1]
    pred_loss = F.mse_loss(reward_pred, reward_truths)

    kl_loss = -0.5 * torch.mean(
        1 + w_log_std - w_mean.pow(2) - torch.exp(w_log_std)
    )
    total_loss = pred_loss + kl_weight * kl_loss
    return total_loss, pred_loss.item(), kl_loss.item()

# —— 训练函数 —— 
# —— 你的 EncoderDecoderModel & Dataset 定义同前 ——


def train_model(model: nn.Module,
                train_loader: DataLoader,
                epochs: int = 20,
                lr: float = 1e-3,
                kl_weight: float = 0.1,
                device: str = 'cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_pred = 0.0
        running_kl   = 0.0

        # 用 tqdm 包裹 train_loader，显示每个 batch 的进度和实时 loss
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in loop:
            batch = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss, pl, kl = compute_vae_loss(model, batch, kl_weight)
            loss.backward()
            optimizer.step()

            # 累积统计
            running_loss += loss.item() * batch[0].size(0)
            running_pred += pl * batch[0].size(0)
            running_kl   += kl * batch[0].size(0)

            # 更新进度条后缀，显示当前 batch 的 loss
            loop.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pred': f"{pl:.4f}",
                'kl':   f"{kl:.4f}"
            })

        n = len(train_loader.dataset)
        avg_loss = running_loss / n
        avg_pred = running_pred / n
        avg_kl   = running_kl / n
        print(f"Epoch {epoch:02d} summary — "
              f"Loss: {avg_loss:.4f}, Pred: {avg_pred:.4f}, KL: {avg_kl:.4f}")
    torch.save(model.state_dict(), "fre_model.pt") 
        
        
def sample_point():
    """
    在 x ∈ [-1.95, 1.95] 上均匀采样：
      - 如果 x ∈ [-0.05, 0.05]，则 y ∈ [0, 1.95]
      - 否则       y ∈ [-1.95, 1.95]
    返回 (x, y)。
    """
    x = np.random.uniform(-1.95, 1.95)
    if -0.05 <= x <= 0.05:
        y = np.random.uniform(0.0, 1.95)
    else:
        y = np.random.uniform(-1.95, 1.95)
    return x, y

def sample_theta():
    """
    在 θ ∈ [-π, π] 上均匀采样。
    返回 θ。
    """
    return np.random.uniform(-np.pi, np.pi)

def sample_rewards(num_episodes, steps_per_episode):
    """
    在 reward_functions 模块中定义的函数上均匀采样。
    返回 (x, y, theta, reward)。
    """

    
    
    pairs = np.zeros((num_episodes, steps_per_episode, 4), dtype=np.float32)
    
    for i in range(num_episodes):
        for j in range(steps_per_episode):
            x, y = sample_point()
            theta = sample_theta()
            # 随机选择一个奖励函数
            reward_func = np.random.choice([
                reward_functions.single_destination,
                reward_functions.two_destinations,
                reward_functions.move_up,
                reward_functions.move_right,
                reward_functions.stronger_storm_better_price,
                reward_functions.butt_scraper
            ])
            reward = reward_func(x, y, theta) / 200
            pairs[i, j, :] = (x, y, theta, reward)
    
    return pairs
    
    

if __name__ == "__main__":
    # 生成假数据
    num_samples, N, state_dim = 500000, 10, 3
    data = sample_rewards(num_samples, N)
    dataset = RewardPairDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EncoderDecoderModel(
        state_dim=state_dim,
        num_discrete_embeddings=21,
        decoder_output_dim=10,   # 比如预测一个标量 reward
        reward_min=-2.0,
        reward_max=5.0
    ).to('cuda')
    train_model(model, loader, epochs=10, lr=1e-3, kl_weight=0.5, device='cuda')
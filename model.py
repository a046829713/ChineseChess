import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from config import GameConfig
import torch
import numpy as np
import time
# 簡單的 Memory 類別，用來儲存訓練數據
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.eaten_states =[]
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.turns = []
        self.masks = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.eaten_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.turns[:]
        del self.masks[:]


# 1. 定義殘差塊 (Residual Block)
# 這是讓網路可以疊深而不退化的關鍵
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection (殘差連接)
        out = F.relu(out)
        return out

# 2. 主模型架構
class ActorCritic(nn.Module):
    def __init__(self, cfg=GameConfig()):
        super(ActorCritic, self).__init__()
        self.cfg = cfg
        
        # --- 設定參數 ---
        self.board_h = 4
        self.board_w = 8
        self.embed_dim = 64     # 增加 Embedding 維度
        self.conv_channels = 64 # 卷積通道數
        self.num_res_blocks = 3 # 殘差塊數量 (越多越深，但也越慢)
        
        # --- 1. 輸入層 ---
        # 棋子 Embedding: 0-14 + 15(Hidden) -> 64維
        self.embedding = nn.Embedding(16, self.embed_dim)
        
        # 初始卷積層 (將 Embedding 轉為特徵圖)
        self.conv_input = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU()
        )
        
        # --- 2. 骨幹網路 (Backbone) ---
        # 堆疊殘差塊
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.conv_channels) for _ in range(self.num_res_blocks)]
        )
        
        # --- 3. 處理 Turn (回合/顏色) ---
        self.turn_embedding = nn.Embedding(2, 16)
        
        # --- 4. 全連接層準備 ---
        # CNN 輸出扁平化後的維度: 64通道 * 4高 * 8寬 = 2048
        self.flatten_dim = self.conv_channels * self.board_h * self.board_w
        
        
        # 加上 Turn 的 embedding 維度
        # turn 16 , no_capture_count 1 eaten_pieces_count 16
        self.fc_input_dim = self.flatten_dim + 16 + 1 + 16
        
        self.fc_shared = nn.Linear(self.fc_input_dim, 512)
        
        # --- 5. 輸出頭 (Heads) ---
        # Action Head (Actor)
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.cfg.TOTAL_ACTIONS)
        )
        
        # Value Head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, turn, eaten_state):
        # state shape: (Batch, 32) -> Flattened integers
        
        # 1. 預處理 State
        state = state.clone().detach().long()
        state[state == -1] = 15 # 處理 Hidden
        
        # Reshape 回 2D 棋盤: (Batch, 4, 8)
        # 注意: 你的 Config 必須確保 NUM_PIECES = 32 = 4*8
        batch_size = state.size(0)
        state_2d = state.view(batch_size, self.board_h, self.board_w) 

        # 2. Embedding + Transpose
        # Embed 輸出: (Batch, 4, 8, Embed_Dim)
        x = self.embedding(state_2d)
        # Permute 為 Conv2d 需要的格式: (Batch, Channels, Height, Width)
        x = x.permute(0, 3, 1, 2) 
        
        # 3. CNN Backbone
        x = self.conv_input(x)
        x = self.res_blocks(x)
        
        
        # 4. Flatten
        x = x.flatten(start_dim=1)
  
        # 5. 加入 Turn 資訊
        if isinstance(turn, int):
            turn = torch.tensor([turn], device=state.device)

        if turn.dim() == 1:
             if turn.size(0) != batch_size:
                 turn = turn.expand(batch_size)
        

        # # 拼接 (Board Features + Turn Features)
        x = torch.cat([x, self.turn_embedding(turn)], dim=1)
        x = torch.cat([x, eaten_state], dim=1)
        
        # 6. Shared FC
        x = F.relu(self.fc_shared(x))
        
        # 7. Outputs
        action_logits = self.action_head(x)
        state_values = self.value_head(x)
        
        return action_logits, state_values

class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # [修正 Bug 5] 將模型搬到正確的 device 上
        self.policy = ActorCritic(self.cfg).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.LEARNING_RATE)
        self.policy_old = ActorCritic(self.cfg).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, turn, mask, eaten_state):
        state_t = torch.LongTensor(state).unsqueeze(0).to(self.device)
        turn_t = torch.LongTensor([turn]).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        # [修正 Bug 3] 使用 FloatTensor (與 Memory 中的 FloatTensor 一致)
        eaten_state_t = torch.FloatTensor(eaten_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.policy_old(state_t, turn_t, eaten_state_t)
            logits[~mask_t] = -float('inf')
            dist = Categorical(logits=logits)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item()


    def evaluate_action(self, state, turn, mask, eaten_state):
        """
            評估模式：不進行隨機抽樣，直接選擇機率最大的合法動作。
        """
        state_t = torch.LongTensor(state).unsqueeze(0).to(self.device)
        turn_t = torch.LongTensor([turn]).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        # [修正 Bug 3] 使用 FloatTensor (與 Memory 中的 FloatTensor 一致)
        eaten_state_t = torch.FloatTensor(eaten_state).unsqueeze(0).to(self.device)

        # 切換到評估模式 (影響 BatchNorm/Dropout 等)
        self.policy.eval()

        with torch.no_grad():
            # 評估時直接用最新的 policy 網路
            logits, _ = self.policy(state_t, turn_t, eaten_state_t)
            
            # 一樣要 Mask 掉不合法的動作
            logits[~mask_t] = -float('inf')
            
            # 使用 argmax 取得最大值的索引 (這就是機率最高的動作)
            action = torch.argmax(logits, dim=1)

        # 恢復為訓練模式
        self.policy.train()
        return action.item()


    def update(self, memory):
        """
            正規 PPO2 更新 (包含 GAE, 優勢正規化, 價值裁剪)


        """
        if not memory.rewards: 
            return {}

        # --- 1. 準備舊資料並計算舊的價值 (No Grad) ---
        with torch.no_grad():
            old_states = torch.stack(memory.states).to(self.device)
            old_turns = torch.tensor(memory.turns, dtype=torch.long).to(self.device)
            old_eaten_states = torch.stack(memory.eaten_states).to(self.device)
            old_actions = torch.stack(memory.actions).to(self.device)
            old_logprobs = torch.stack(memory.logprobs).to(self.device)
            old_masks = torch.stack(memory.masks).to(self.device)
            
            _, old_state_values = self.policy(old_states, old_turns, old_eaten_states)
            # 壓平並傳到 CPU 以利順序迴圈計算
            old_state_values = old_state_values.squeeze(-1)
            old_values_cpu = old_state_values.cpu()

        # --- 2. 計算 GAE (Generalized Advantage Estimation) ---
        advantages = []
        gae = 0
        gae_lambda = getattr(self.cfg, 'GAE_LAMBDA', 0.95)

        for i in reversed(range(len(memory.rewards))):
            if memory.is_terminals[i] or i == len(memory.rewards) - 1:
                next_value = 0.0
            else:
                next_value = old_values_cpu[i + 1].item()

            delta = memory.rewards[i] + self.cfg.GAMMA * next_value - old_values_cpu[i].item()
            gae = delta + self.cfg.GAMMA * gae_lambda * gae * (not memory.is_terminals[i])
            advantages.append(gae)
            
        advantages.reverse()
        
        # 轉為 GPU Tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # --- 3. 計算 Returns (Critic 的目標值) 並正規化 Advantage ---
        # Returns = Advantages + Old Values
        returns = advantages + old_state_values 

        # 正規化 Advantage (在 Epoch 迴圈外只做一次)
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 4. PPO 迭代更新 (K_EPOCHS) ---
        diagnostics = {}

        for _ in range(self.cfg.K_EPOCHS):
            # Forward pass (計算新的 logits 和 values)
            logits, state_values = self.policy(old_states, old_turns, old_eaten_states)
            logits[~old_masks] = -float('inf')  # Apply Mask
            state_values = state_values.squeeze(-1)
            
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # --- Actor Loss ---
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1-self.cfg.EPS_CLIP, 1+self.cfg.EPS_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # --- Critic Loss (導入 PPO2 標準的 Value Clipping) ---
            # 這是為了防止 Critic 在一個 epoch 內更新幅度過大
            value_loss_unclipped = (state_values - returns) ** 2
            state_values_clipped = old_state_values + torch.clamp(
                state_values - old_state_values, 
                -self.cfg.EPS_CLIP, 
                self.cfg.EPS_CLIP
            )
            value_loss_clipped = (state_values_clipped - returns) ** 2
            # 係數改回標準的 0.5 (因為 GAE 已經大幅降低方差，不需要再壓到 0.25)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # --- 總 Loss ---
            entropy_loss = -0.02 * dist_entropy.mean()
            loss = actor_loss + critic_loss + entropy_loss
            
            # --- Backpropagation ---
            self.optimizer.zero_grad()
            loss.backward()
            # self.get_weight_stats()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            # --- KL 早停機制與診斷指標 ---
            with torch.no_grad():
                kl_div = (old_logprobs - new_logprobs).mean().item()
                clip_fraction = ((ratios - 1.0).abs() > self.cfg.EPS_CLIP).float().mean().item()
                
                diagnostics = {
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy': dist_entropy.mean().item(),
                    'total_loss': loss.item(),
                    'clip_fraction': clip_fraction,
                    'kl_divergence': kl_div,
                    'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'value_pred_mean': state_values.mean().item(),
                    'advantage_mean': advantages.mean().item(),
                    'ratio_mean': ratios.mean().item(),
                }
                
                if hasattr(self.cfg, 'KL_TARGET') and self.cfg.KL_TARGET is not None:
                    if abs(kl_div) > self.cfg.KL_TARGET * 1.5:
                        break  # 觸發早停
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        return diagnostics


    def get_weight_stats(self):
        """取得模型各層權重的統計資訊，用於診斷"""
        stats = {}
        for name, param in self.policy.named_parameters():
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'norm': param.data.norm().item(),
            }
        print(stats)
        return stats


    # --- 儲存模型 ---
    def save_model(self, path):
        try:
            torch.save(self.policy.state_dict(), path)
        except Exception as e:
            print(f"Error saving model: {e}")

    # --- 載入模型 ---
    def load_model(self, path):
        try:
            # [修正] 使用 self.device 確保在正確的裝置上載入
            state_dict = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            self.policy_old.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

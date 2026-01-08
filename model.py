import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
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
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.turns = []
        self.masks = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
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
        self.fc_input_dim = self.flatten_dim + 16 
        
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

    def forward(self, state, turn):
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
        
        
        
        turn_embed = self.turn_embedding(turn).view(batch_size, -1)

        # 拼接 (Board Features + Turn Features)
        x = torch.cat([x, turn_embed], dim=1)
        
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
        self.policy = ActorCritic(self.cfg)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.LEARNING_RATE)
        self.policy_old = ActorCritic(self.cfg)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, turn, mask):
        state_t = torch.LongTensor(state).unsqueeze(0)
        turn_t = torch.LongTensor([turn])
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        

        with torch.no_grad():
            logits, _ = self.policy_old(state_t, turn_t)
            logits[~mask_t] = -float('inf')
            dist = Categorical(F.softmax(logits, dim=1))
            action = dist.sample()
        return action.item(), dist.log_prob(action).item()


    def evaluate_action(self, state, turn, mask):
        """
            評估模式：不進行隨機抽樣，直接選擇機率最大的合法動作。
        """
        state_t = torch.LongTensor(state).unsqueeze(0).to(self.device)
        turn_t = torch.LongTensor([turn]).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        # 切換到評估模式 (影響 BatchNorm/Dropout 等)
        self.policy.eval()

        with torch.no_grad():
            # 評估時直接用最新的 policy 網路
            logits, _ = self.policy(state_t, turn_t)
            
            # 一樣要 Mask 掉不合法的動作
            logits[~mask_t] = -float('inf')
            
            # 使用 argmax 取得最大值的索引 (這就是機率最高的動作)
            action = torch.argmax(logits, dim=1)

        # 恢復為訓練模式
        self.policy.train()
        return action.item()



    def update(self, memory):
        if not memory.rewards: 
            return
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.cfg.GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.stack(memory.states).detach()
        old_turns = torch.tensor(memory.turns, dtype=torch.long) # 處理 Turn
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_masks = torch.stack(memory.masks).detach() # 處理 Mask

        for _ in range(self.cfg.K_EPOCHS):
            logprobs, state_values = self.policy(old_states, old_turns)
            logprobs[~old_masks] = -float('inf') # Apply Mask
            
            dist = Categorical(F.softmax(logprobs, dim=1))
            new_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            
            surr2 = torch.clamp(ratios, 1-self.cfg.EPS_CLIP, 1+self.cfg.EPS_CLIP) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.05*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            # self.checkgrad(self.policy)

            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    # --- 新增：儲存模型 ---
    def save_model(self, path):
        try:
            torch.save(self.policy.state_dict(), path)
        except Exception as e:
            print(f"Error saving model: {e}")

    # --- 新增：載入模型 ---
    def load_model(self, path):
        try:
            # map_location 確保在 CPU/GPU 之間切換不會報錯
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(state_dict)
            self.policy_old.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def checkgrad(self,net):
        # 打印梯度統計數據
        for name, param in net.named_parameters():
            if param.grad is not None:
                print(
                    f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}")

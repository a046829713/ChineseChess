import math
import numpy as np
import torch
from torch.distributions import Categorical
import time

class Node:
    def __init__(self, prior_prob):
        self.P = prior_prob  # 神經網路給予的先驗機率 P(s, a)
        self.N = 0           # 訪問次數 N(s, a)
        self.W = 0.0         # 總價值 W(s, a)
        self.Q = 0.0         # 平均價值 Q(s, a) = W / N
        self.children = {}   # 儲存子節點 {action: Node}

    def expand(self, action_probs):
        """
            將神經網路輸出的機率展開成子節點
        
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(prior_prob=prob)

    def get_ucb(self, c_puct):
        """
            計算 PUCT 公式中的 UCB 分數

            Upper Confidence Bound

            PUCT = Q(s, a) + C(s) \times P(s, a) \times \frac{\sqrt{N}}{1 + n_a}
        """
        # 父節點的總訪問次數 N(s)
        parent_N = sum(child.N for child in self.children.values())
        
        # 避免除以零
        if self.N == 0:
            # 如果還沒被探索過，Q 視為 0，完全依賴先驗機率 P
            return c_puct * self.P * math.sqrt(parent_N + 1e-8)
        
        # PUCT 公式
        ucb = self.Q + c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)
        return ucb

    def show_node(self,level = 0):
        print("test going")
        level =level +1
        print("right now level:",level)


        print(len(self.children))
        for children_node_info in self.children.items():
            children_node = children_node_info[1]
            print("level:",level , children_node)
            if children_node.children:                
                children_node.show_node(level)

    def count_all_node(self):
        count = 0
        
        for children_node_info in self.children.items():
            children_node = children_node_info[1]
            count +=1

            if children_node.children:                
                count += children_node.count_all_node()

        
        return count
    
    def get_max_depth(self):
        """計算這棵搜尋樹目前最深走到了第幾層"""       
        if not self.children:
            return 1

        return 1 + max(child.get_max_depth() for child in self.children.values())



class MCTS:
    def __init__(self, policy_net, cfg, device, num_simulations=50, c_puct=1.5):
        self.policy_net = policy_net
        self.cfg = cfg
        self.num_simulations = num_simulations # 筆電算力建議先設 50~100
        self.c_puct = c_puct
        self.device = device

    def _prepare_tensors(self,state, eaten_state):
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        eaten_state_t = torch.FloatTensor(eaten_state).unsqueeze(0).to(self.device)
        return state_t, eaten_state_t
        
    def search(self, initial_env, state_tensor, eaten_state_tensor):
        """
        執行 MCTS 搜尋，回傳每個動作的 MCTS 改善後機率分佈
        """
        root = Node(prior_prob=1.0)
        
        # --- 1. 展開根節點 ---
        self.policy_net.eval()
        with torch.no_grad():
            logits, _ = self.policy_net(state_tensor, eaten_state_tensor)
            mask = initial_env.get_legal_actions(initial_env.turn)
            
            # Mask 非法動作並轉為機率
            logits_masked = logits.clone().squeeze(0)
            logits_masked[~torch.BoolTensor(mask).to(logits.device)] = -float('inf')
            probs = torch.softmax(logits_masked, dim=0).cpu().numpy()
        
        # 建立合法的 Action -> Prob 字典來展開 root
        legal_actions = np.where(mask)[0]
        action_probs = {a: probs[a] for a in legal_actions}
        
        
        root.expand(action_probs)


        
        # --- 2. 核心模擬迴圈 ---
        for _ in range(self.num_simulations):
            node = root

  
            # 複製一個虛擬環境來推演 (極度重要！)
            env_clone = initial_env.clone() 
            search_path = [node]

            # (A) Selection: 一直往下選 UCB 最高的子節點，直到遇到葉節點 (Leaf)
            while node.children:

                best_action = max(node.children.keys(), key=lambda a: node.children[a].get_ucb(self.c_puct))
                
                node = node.children[best_action]
                search_path.append(node)
                


                # 在虛擬環境中走這步棋
                next_state_np, reward, done, _ = env_clone.step(best_action, i_episode=0)



            # (B) Expansion & Evaluation: 到了葉節點，呼叫神經網路評估
            if not done:
                # [重要] 這裡需要一個將 numpy 轉為 tensor 的輔助函數，或者直接用 env_clone 的數據
                # 假設 env_clone.get_tensors() 能吐出 state_tensor, eaten_tensor
                # 這裡為求範例簡潔，假設你已經轉好 tensor:
                state, eaten_state = next_state_np
                st, et = self._prepare_tensors(state,eaten_state=eaten_state)
                with torch.no_grad():
                    leaf_logits, leaf_value = self.policy_net(st, et)
                    leaf_value = leaf_value.item() # 取出標量 (-1 到 +1)
                    
                    leaf_mask = env_clone.get_legal_actions(env_clone.turn)
                    leaf_logits_masked = leaf_logits.squeeze(0)
                    leaf_logits_masked[~torch.BoolTensor(leaf_mask).to(leaf_logits.device)] = -float('inf')
                    leaf_probs = torch.softmax(leaf_logits_masked, dim=0).cpu().numpy()

                # 展開這個葉節點
                leaf_legal_actions = np.where(leaf_mask)[0]
                leaf_action_probs = {a: leaf_probs[a] for a in leaf_legal_actions}
                node.expand(leaf_action_probs)
            else:
                # 如果遊戲在推演中結束了，直接使用真實 Reward (+1 或 -1)
                leaf_value = reward 

            # (C) Backpropagation: 把 Value 往回傳遞
            # 注意：因為暗棋是零和遊戲，父節點與子節點的視角是相反的，所以 value 要乘 -1
            for n in reversed(search_path):
                n.W += leaf_value
                n.N += 1
                n.Q = n.W / n.N
                leaf_value = -leaf_value 

        
        print(f"本次 MCTS 最大搜尋深度: {root.get_max_depth()}")
        
        
        
        
        # --- 3. 根據訪問次數計算最終的輸出機率 ---
        # 訪問次數越多的動作，代表 MCTS 認為越好
        action_counts = np.zeros(initial_env.cfg.TOTAL_ACTIONS)
        for action, child in root.children.items():
            action_counts[action] = child.N
            
        # 加上溫度參數 (Temperature) 控制探索 (這裡簡化為直接算比例)
        mcts_probs = action_counts / np.sum(action_counts)
        return mcts_probs
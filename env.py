import numpy as np
import random
from config import GameConfig
import time
from collections import deque
import torch


class DarkChessEnv:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg else GameConfig()
        self.board = np.full(self.cfg.NUM_PIECES, self.cfg.HIDDEN)
        self.actual_board = np.zeros(self.cfg.NUM_PIECES, dtype=int)
        


        # --- [新增] AlphaZero 風格的狀態歷史堆疊 ---
        self.history_length = 8
        self.state_buffer = deque(maxlen=self.history_length)

        self.turn = 0 # 0: Player 1, 1: Player 2
        self.my_color = self.cfg.COLOR_UNKNOWN
        self.game_over = False
        self.winner = None
        self.no_capture_count = 0
        self.state_history = {} # 記錄局面出現次數 (Board Tuple + Turn) -> Count
        
        # 初始化真實棋盤 (洗牌)
        pieces = []
        pieces.extend([1]*1 + [2]*2 + [3]*2 + [4]*2 + [5]*2 + [6]*2 + [7]*5)
        pieces.extend([8]*1 + [9]*2 + [10]*2 + [11]*2 + [12]*2 + [13]*2 + [14]*5)
        random.shuffle(pieces)

        self.actual_board = np.array(pieces)

        self.eaten_pieces_count = [0] * 15
        self.state_key = tuple()

    def reset(self):
        self.__init__(self.cfg)
        self._update_history() # 記錄初始局面
        return self.get_state()

    def _update_history(self):
        """更新並回傳當前局面的重複次數"""
        # 將棋盤轉為 tuple (不可變，可作為 dict key)，加上當前輪次
        state_key = (tuple(self.board), self.turn)
        self.state_history[state_key] = self.state_history.get(state_key, 0) + 1
        return self.state_history[state_key]

    # def get_state(self):
    #     # --- [修改] 將 8 步的歷史轉換為 NumPy 陣列返回 ---
    #     # 回傳的 stacked_board shape 為 (8, 32)
    #     stacked_board = np.array(self.state_buffer)
        
    #     return stacked_board, self.turn, self.eaten_pieces_count + [self.no_capture_count /10 ] + [self.state_history.get(self.state_key, 0)]


    def get_state(self):
        """
        將 1D board (長度 32) 轉換為 (18, 4, 8) 的 One-Hot Float Tensor
        """
        # 1. 將 1D 棋盤折疊成 2D (4列, 8欄)
        board_2d = self.board.reshape(4, 8)
        
        # 2. 定義所有可能的棋子 ID (共 16 種狀態)
        # 確保順序固定，神經網路才能認得哪個通道代表什麼
        piece_ids = [self.cfg.EMPTY, self.cfg.HIDDEN] + list(range(1, 15))
        
        # 將 piece_ids 轉為形狀 (16, 1, 1) 以便利用 NumPy 的廣播機制
        piece_ids_array = np.array(piece_ids).reshape(16, 1, 1)
        
        # 3. 極速 One-Hot 轉換核心 (消除 Python 迴圈)
        # board_2d 的 shape 是 (4, 8)
        # piece_ids_array 的 shape 是 (16, 1, 1)
        # 兩者判斷相等 (==) 後，會自動擴展並回傳 shape 為 (16, 4, 8) 的布林矩陣
        one_hot_planes = (board_2d == piece_ids_array).astype(np.float32)
        
        # 4. 建立全局特徵通道 (Global Features)
        # 建立形狀為 (2, 4, 8) 的矩陣來存放回合與計數器
        global_planes = np.zeros((2, 4, 8), dtype=np.float32)
        
        # 通道 16: 回合資訊 (紅方=1, 黑方=0)
        # 如果 self.turn 為 0 代表 Player 1，你可以依照你的顏色邏輯微調
        global_planes[0].fill(1.0 if self.turn == 0 else 0.0) 
        
        # 通道 17: 正規化的無吃子步數 (假設 60 步逼和)
        global_planes[1].fill(self.no_capture_count / 60.0)
        
        # 5. 將 One-Hot 棋盤與全局特徵在通道維度 (axis=0) 拼接
        # 最終 shape: (18, 4, 8)
        state_tensor_np = np.concatenate([one_hot_planes, global_planes], axis=0)
        
        # 6. 轉為 PyTorch Tensor 供神經網路使用
        return torch.from_numpy(state_tensor_np) ,self.eaten_pieces_count


    def _pos_to_coord(self, pos):
        return divmod(pos, self.cfg.BOARD_WIDTH)

    def _is_my_piece(self, piece, player_idx):
        """檢查棋子是否屬於當前玩家"""
        if piece == self.cfg.HIDDEN or piece == self.cfg.EMPTY:
            return False
        
        # 如果還沒決定顏色，任何翻開的棋子在第一次行動前都不屬於誰(不能移動)，只能翻牌
        # 但這裡我們只檢查顏色歸屬
        if self.my_color == self.cfg.COLOR_UNKNOWN:
            return False
            
        current_color = self.my_color if player_idx == 0 else (1 - self.my_color)
        is_red = (piece in self.cfg.RED_PIECES)
        if current_color == self.cfg.COLOR_RED and is_red: return True
        if current_color == self.cfg.COLOR_BLACK and not is_red: return True
        return False

    def get_legal_actions(self, player_idx):
        """
        回傳一個長度為 1024 的 bool mask
        Action ID = source_pos * 32 + target_pos
        """
        mask = np.zeros(self.cfg.TOTAL_ACTIONS, dtype=bool)
        
        for src in range(self.cfg.NUM_PIECES):
            # 1. 檢查翻牌 (Source == Target)
            if self.board[src] == self.cfg.HIDDEN:
                action_id = src * self.cfg.NUM_PIECES + src
                mask[action_id] = True
                continue # 蓋著的牌不能移動
            
            # 2. 檢查移動/吃子 (Source != Target)
            # 必須是自己的棋子才能移動
            if self._is_my_piece(self.board[src], player_idx):
                piece = self.board[src]
                
                # 針對這個棋子，掃描所有可能的目標位置
                # 為了效能，我們只掃描十字線上的格子 (因為暗棋只能走直線)
                # 或是簡單點，掃描所有格子交給 check_move_rule 判斷 (程式碼較乾淨，Python迴圈較慢)
                # 這裡採取掃描所有格子的方式以確保邏輯統一
                for dst in range(self.cfg.NUM_PIECES):
                    if src == dst:
                        continue
                    
                    if self._check_move_rule(src, dst, piece):
                        action_id = src * self.cfg.NUM_PIECES + dst
                        mask[action_id] = True
        return mask

    def _check_move_rule(self, src, dst, piece):
        """
        核心規則邏輯：判斷從 src 移動到 dst 是否合法
        包含：移動距離、炮的邏輯、目標格狀態
        """
        target_piece = self.board[dst]
        
        # 目標不能是蓋著的牌
        if target_piece == self.cfg.HIDDEN:
            return False
        
        sr, sc = self._pos_to_coord(src)
        dr, dc = self._pos_to_coord(dst)
        
        # 只能走直線
        if sr != dr and sc != dc:
            return False
            
        dist = abs(sr - dr) + abs(sc - dc)
        
        # 計算路徑上的障礙物數量 (不含 src 和 dst)
        obstacles = 0
        if sr == dr: # 同列
            min_c, max_c = min(sc, dc), max(sc, dc)
            for c in range(min_c + 1, max_c):
                if self.board[sr * self.cfg.BOARD_WIDTH + c] != self.cfg.EMPTY:
                    obstacles += 1
        else: # 同行
            min_r, max_r = min(sr, dr), max(sr, dr)
            for r in range(min_r + 1, max_r):
                if self.board[r * self.cfg.BOARD_WIDTH + sc] != self.cfg.EMPTY:
                    obstacles += 1

        # --- 炮/包 (Cannon) 的特殊邏輯 ---
        if piece == 6 or piece == 13:
            if obstacles == 0:
                # 路徑無障礙：只能走一格，且目標必須是空的 (移動)
                # 標準暗棋炮不能跳著移動到空地，也不能滑行
                if dist == 1 and target_piece == self.cfg.EMPTY:
                    return True
                return False
            elif obstacles == 1:
                # 有一個障礙 (炮架)：目標必須是敵人 (吃子)
                if target_piece != self.cfg.EMPTY and self._is_enemy(piece, target_piece):
                    return True
                return False
            else:
                # 障礙物 > 1：無法跳過
                return False
        
        # --- 非炮類棋子 (一般邏輯) ---
        else:
            # 只能走一格
            if dist != 1:
                return False
            
            # 檢查障礙物 (其實 dist=1 障礙物一定是0，但保留邏輯)
            if obstacles > 0: 
                return False
                
            if target_piece == self.cfg.EMPTY:
                return True # 移動
            else:
                # 吃子判定
                return self._can_eat(piece, target_piece)

    def _is_enemy(self, my_piece, target_piece):
        """判斷目標是否為敵人"""
        if target_piece == self.cfg.EMPTY or target_piece == self.cfg.HIDDEN: return False
        is_me_red = (my_piece in self.cfg.RED_PIECES)
        is_target_red = (target_piece in self.cfg.RED_PIECES)
        return is_me_red != is_target_red

    def _is_piece_attacking(self, src):
        """檢查位於 src 的棋子是否正在攻擊任何敵人"""
        piece = self.board[src]
        if piece == self.cfg.EMPTY or piece == self.cfg.HIDDEN:
            return False
            
        # 掃描所有格子，看是否能吃子
        for dst in range(self.cfg.NUM_PIECES):
            if src == dst: continue
            
            target_piece = self.board[dst]
            if target_piece == self.cfg.EMPTY or target_piece == self.cfg.HIDDEN:
                continue
                
            # 必須是敵人
            if not self._is_enemy(piece, target_piece):
                continue
                
            # 必須符合移動/吃子規則
            # 注意: _check_move_rule 裡已經包含了 _can_eat 的邏輯 (針對非炮類)
            # 但針對炮類，_check_move_rule 負責路徑和炮架，_can_eat 負責階級
            # 為了保險，我們這兩個都檢查
            if self._check_move_rule(src, dst, piece):
                # 修正：如果是炮(6 或 13)，只要 move_rule 過了就算攻擊，不檢查階級
                if piece == 6 or piece == 13:
                    return True
                
                # 所以為了統一，我們手動再檢查一次 _can_eat
                if self._can_eat(piece, target_piece):
                    return True
        return False

    def _can_eat(self, attacker, victim):
        """階級吃子規則"""
        if not self._is_enemy(attacker, victim):
            return False
            
        # 轉換為階級 1-7
        a_rank = attacker if attacker <= 7 else attacker - 7
        v_rank = victim if victim <= 7 else victim - 7
        
        # 炮(6) 只要滿足炮架邏輯，大小通吃 (除了不能直接吃，但這由 move_rule 處理)
        # 在 _check_move_rule 裡，非炮棋子才會呼叫這個，所以這裡只需處理非炮
        # 但如果炮是 target (被吃)，則按階級
        
        # 帥(1) vs 卒(7)
        if a_rank == 1 and v_rank == 7: return False
        if a_rank == 7 and v_rank == 1: return True
        return a_rank <= v_rank


    def _update_eaten_pieces_count(self, eaten_piece):
        self.eaten_pieces_count[eaten_piece] += 1

    def step(self, action, i_episode):
        # 解析動作
        src = action // self.cfg.NUM_PIECES
        dst = action % self.cfg.NUM_PIECES
 
        reward = 0
        done = False
        info = {}
        
        # --- 翻牌邏輯 ---
        if src == dst:
            if self.board[src] != self.cfg.HIDDEN:
                return self.get_state(), self.cfg.REWARD_INVALID, self.game_over, {"error": "Invalid Flip"}
            
            piece = self.actual_board[src]
            self.board[src] = piece
            reward = self.cfg.REWARD_FLIP
            self.no_capture_count += 1
            
            # 決定顏色
            if self.my_color == self.cfg.COLOR_UNKNOWN:
                if piece in self.cfg.RED_PIECES:
                    self.my_color = self.cfg.COLOR_RED
                else:
                    self.my_color = self.cfg.COLOR_BLACK
        
        # --- 移動/吃子邏輯 ---
        else:
            piece = self.board[src]
            
            # 再次檢查合法性 (防止 Model 輸出非法動作)
            # 在 RL 訓練中，通常會先 mask 掉非法動作，但這裡做個保險
            if not self._is_my_piece(piece, self.turn) or not self._check_move_rule(src, dst, piece):
                return self.get_state(), self.cfg.REWARD_INVALID, self.game_over, {"error": "Invalid Move"}

            target_piece = self.board[dst]
            
            if target_piece == self.cfg.EMPTY:
                # 移動
                self.board[dst] = piece
                self.board[src] = self.cfg.EMPTY
                reward = self.cfg.REWARD_MOVE
                self.no_capture_count += 1
            else:
                # 吃子
                self._update_eaten_pieces_count(self.board[dst])
                reward = self.cfg.PIECE_VALUES[self.board[dst]]

                self.board[dst] = piece
                self.board[src] = self.cfg.EMPTY
                

                info["Eaten_reward"] = reward * -1
                self.no_capture_count = 0

        # --- 檢查重複局面 (長捉/長將禁手判斷) ---
        # 預判下一個狀態 (因為 step 結尾才會切換 turn，這裡先模擬切換後的狀態 key)
        next_turn = 1 - self.turn
        self.state_key = (tuple(self.board), next_turn)
        
        self.state_history[self.state_key] = self.state_history.get(self.state_key, 0) + 1
        
        if self.state_history[self.state_key] >= 3:
            # 如果同一局面重複 3 次
            # 判斷是「長捉/長將」(Perpetual Chase) 還是「閒著/雙方互走」(Mutual Repetition)
            
            # 剛剛移動的棋子位置是 dst
            # 檢查這步棋是否造成了攻擊 (捉/將)
            if self._is_piece_attacking(dst):
                print(f"目前局數: {i_episode} 長捉禁手")
                 # 長捉/長將 -> 判負 (禁手)
                self.game_over = True
                self.winner = next_turn # 對手獲勝 (1 - self.turn)
            else:
                print(f"目前局數: {i_episode} 無意義和局")
                # 閒著/無意義重複 -> 和局
                self.game_over = True
                self.winner = None # 和局
                
        # --- 判定勝負 ---
        self._check_game_over(i_episode)
        reward = self._adjust_reward_for_endgame(reward)
        self.turn = 1 - self.turn

        
        return self.get_state(), reward, self.game_over, info

    def _check_game_over(self, i_episode:int):
        visible_red = np.any(np.isin(self.board, self.cfg.RED_PIECES))
        visible_black = np.any(np.isin(self.board, self.cfg.BLACK_PIECES))
        hidden_count = np.sum(self.board == self.cfg.HIDDEN)
        
        # 簡單勝負判定：如果場上沒有某一色的棋子且沒有蓋牌 -> 輸
        # (完整的暗棋還有逼和規則，這裡簡化處理)
        if hidden_count == 0:                       
            if not visible_red:
                self.winner = 0 if self.my_color == self.cfg.COLOR_BLACK else 1
                self.game_over = True

            elif not visible_black:
                self.winner = 0 if self.my_color == self.cfg.COLOR_RED else 1
                self.game_over = True


        if self.no_capture_count >= 60:
            print(f"目前局數: {i_episode} , 60步沒有吃到")
            self.game_over = True
            self.winner = None # 和局
            


    def clone(self, determinize=True):
        """
        創造一個平行的虛擬環境供 MCTS 推演未來。
        :param determinize: 是否對暗牌進行重新洗牌 (預設 True，用於 MCTS 模擬)
        """
        new_env = DarkChessEnv(self.cfg)
        
        # 1. 複製所有會影響遊戲狀態的變數 (使用 .copy() 避免記憶體共用)
        new_env.board = self.board.copy()
        new_env.turn = self.turn
        new_env.my_color = self.my_color
        new_env.game_over = self.game_over
        new_env.winner = self.winner
        new_env.no_capture_count = self.no_capture_count
        new_env.state_history = self.state_history.copy()
        new_env.eaten_pieces_count = self.eaten_pieces_count.copy()
        # [注意] state_key 也必須複製，否則重複判斷會出錯
        new_env.state_key = self.state_key 

        if not determinize:
            # 如果只是想單純備份環境 (不進行 MCTS 模擬)，直接複製真實底牌
            new_env.actual_board = self.actual_board.copy()
            return new_env

        # ==========================================
        # 核心：Determinization (確定化 / 隨機洗牌)
        # ==========================================
        
        # 步驟 A：定義初始的完整棋子池 (1~14)
        initial_counts = {
            1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 5,  # 紅方: 帥, 仕, 相, 俥, 傌, 炮, 兵
            8: 1, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 5 # 黑方: 將, 士, 象, 車, 馬, 包, 卒
        }
        
        # 步驟 B：統計目前場上「已經翻開」的棋子
        visible_counts = {i: 0 for i in range(1, 15)}
        for piece in self.board:
            if piece != self.cfg.HIDDEN and piece != self.cfg.EMPTY:
                visible_counts[piece] += 1
                
        # 步驟 C：計算「還蓋在底下」的棋子種類與數量
        hidden_pool = []
        for piece_id in range(1, 15):
            # 剩餘數量 = 初始總數 - 場上已亮出的數量 - 已經被吃掉的數量
            remaining = initial_counts[piece_id] - visible_counts[piece_id] - self.eaten_pieces_count[piece_id]
            
            # 防呆保護：避免計數錯誤導致負數
            remaining = max(0, remaining) 
            hidden_pool.extend([piece_id] * remaining)
            
        # 確保我們算出來的暗牌數量，等於棋盤上實際蓋著的數量
        hidden_on_board = np.sum(self.board == self.cfg.HIDDEN)
        if len(hidden_pool) != hidden_on_board:
            # 如果這個 print 觸發，代表你的 eaten_pieces_count 在 step 邏輯中有 bug
            print(f"Warning: 算出的暗牌數({len(hidden_pool)})與盤面實際暗牌數({hidden_on_board})不符！")
            # 強制截斷或補齊 (通常不會發生)
            if len(hidden_pool) > hidden_on_board:
                hidden_pool = hidden_pool[:hidden_on_board]
            else:
                hidden_pool.extend([1] * (hidden_on_board - len(hidden_pool)))

        # 步驟 D：模擬平行宇宙 (洗牌)
        random.shuffle(hidden_pool)
        
        # 步驟 E：重建虛擬環境的 actual_board
        new_actual_board = np.zeros(self.cfg.NUM_PIECES, dtype=int)
        pool_idx = 0
        
        for i in range(self.cfg.NUM_PIECES):
            if self.board[i] == self.cfg.HIDDEN:
                # 遇到蓋著的牌：從洗過的牌堆中發一張給它
                new_actual_board[i] = hidden_pool[pool_idx]
                pool_idx += 1
            elif self.board[i] != self.cfg.EMPTY:
                # 遇到已翻開的牌：底牌就是它自己
                new_actual_board[i] = self.board[i]
            else:
                # 遇到空地
                new_actual_board[i] = self.cfg.EMPTY
                
        new_env.actual_board = new_actual_board
        
        return new_env

    def _adjust_reward_for_endgame(self, reward):
        if self.game_over:
            if self.winner is None :
                if self.no_capture_count >= 60:
                    # 可能真的無法吃到 所以才拖到60步
                    reward += self.cfg.REWARD_DRAW
                    print("60步獎勵: ",reward)
                else:
                    # 無意義和棋
                    reward += self.cfg.REWARD_LOSS_DRAW
            else:
                # 判斷當前行動者是否獲勝
                # 注意：step 函式末尾才切換 turn，所以這裡的 self.turn 還是當前行動者
                if self.winner == self.turn:
                    reward += self.cfg.REWARD_WIN
                else:
                    reward += self.cfg.REWARD_LOSE
        
        return reward

# 測試代碼
if __name__ == "__main__":
    index = 0
    while True:
        print("目前局數:",index)
        env = DarkChessEnv()
        s, t, _= env.reset()
        # print("Initial Board (Hidden):")
        
        
        # 隨機測試
        while True:
            print(env.board)
            print("*"*120)


            mask = env.get_legal_actions(env.turn)
            legal_indices = np.where(mask)[0]
            
            action = np.random.choice(legal_indices)
            state, reward, done, _ = env.step(action)
            
            print("是否完成:",done)
            if done:
                print("Game Over")
                break
        
        index += 1
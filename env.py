import numpy as np
import random
from config import GameConfig
import time




class DarkChessEnv:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg else GameConfig()
        self.board = np.full(self.cfg.NUM_PIECES, self.cfg.HIDDEN)
        self.actual_board = np.zeros(self.cfg.NUM_PIECES, dtype=int)
        
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

    def get_state(self):
        return self.board.copy(), self.turn

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

    def step(self, action):
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
                self.board[dst] = piece
                self.board[src] = self.cfg.EMPTY
                reward = self.cfg.REWARD_EAT
                self.no_capture_count = 0

        # --- 檢查重複局面 (長捉/長將禁手判斷) ---
        # 預判下一個狀態 (因為 step 結尾才會切換 turn，這裡先模擬切換後的狀態 key)
        next_turn = 1 - self.turn
        state_key = (tuple(self.board), next_turn)
        self.state_history[state_key] = self.state_history.get(state_key, 0) + 1
        
        if self.state_history[state_key] >= 3:
            # 如果同一局面重複 3 次，判定當前行動者(造成重複的人)判負 (禁手)
            self.game_over = True
            self.winner = next_turn # 對手獲勝
            reward = self.cfg.REWARD_LOSE # 給予當前玩家懲罰
        
        # --- 判定勝負 ---
        self._check_game_over(reward)
        reward = self._adjust_reward_for_endgame(reward)
        
        self.turn = 1 - self.turn
        return self.get_state(), reward, self.game_over, info

    def _check_game_over(self, current_reward):
        visible_red = np.any(np.isin(self.board, self.cfg.RED_PIECES))
        visible_black = np.any(np.isin(self.board, self.cfg.BLACK_PIECES))
        hidden_count = np.sum(self.board == self.cfg.HIDDEN)
        
        # 簡單勝負判定：如果場上沒有某一色的棋子且沒有蓋牌 -> 輸
        # (完整的暗棋還有逼和規則，這裡簡化處理)
        if hidden_count == 0:
            if not visible_red:
                self.winner = self.cfg.COLOR_BLACK
                self.game_over = True
            elif not visible_black:
                self.winner = self.cfg.COLOR_RED
                self.game_over = True
        
        if self.no_capture_count >= 60:
            self.game_over = True
            self.winner = None # 和局

    def _adjust_reward_for_endgame(self, reward):
        if self.game_over:
            if self.winner is None:
                return self.cfg.REWARD_DRAW
            
            # 判斷當前行動者是否獲勝
            # 注意：step 函式末尾才切換 turn，所以這裡的 self.turn 還是當前行動者
            current_player_color = self.my_color if self.turn == 0 else (1 - self.my_color)
            
            if self.winner == current_player_color:
                return self.cfg.REWARD_WIN
            else:
                return self.cfg.REWARD_LOSE
        return reward

# 測試代碼
if __name__ == "__main__":
    index = 0
    while True:
        print("目前局數:",index)
        env = DarkChessEnv()
        s, t = env.reset()
        # print("Initial Board (Hidden):")
        
        
        # 隨機測試
        while True:
            mask = env.get_legal_actions(env.turn)

            legal_indices = np.where(mask)[0]
            
            if len(legal_indices) == 0:
                print(np.where(mask))
                print(env.board.reshape(env.cfg.BOARD_HEIGHT, env.cfg.BOARD_WIDTH))
                print(env.turn)
                print("No legal moves!")
                time.sleep(1000)
                break
                
            action = np.random.choice(legal_indices)
            state, reward, done, _ = env.step(action)
            
            if done:
                print("Game Over")
                break
        
        index += 1
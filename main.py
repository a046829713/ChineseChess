import tkinter as tk
from tkinter import messagebox
import threading
import time
import copy
from config import GameConfig
from env import DarkChessEnv
from model import PPOAgent
from model import Memory
from diagnostic import TrainingDiagnostics
import torch
import os

class DarkChessGUI:
    def __init__(self, root):
        self.root = root
        self.cfg = GameConfig()
        self.check_file()

    
        self.root.title(f"PPO Dark Chess - {self.cfg.BOARD_WIDTH}x{self.cfg.BOARD_HEIGHT}")
        
        self.env = DarkChessEnv(self.cfg)        
        self.agent = PPOAgent(self.cfg) 

        self.memory_red = Memory()   # 專存紅方的 (s, a, r)
        self.memory_black = Memory() # 專存黑方的 (s, a, r)
        
        # [新增] 訓練診斷工具
        self.diagnostics = TrainingDiagnostics(save_dir=self.cfg.SAVE_PATH)

        self.frame_container = tk.Frame(root)
        self.frame_container.pack()

        c_w = self.cfg.BOARD_WIDTH * self.cfg.CELL_SIZE
        c_h = self.cfg.BOARD_HEIGHT * self.cfg.CELL_SIZE
        self.canvas = tk.Canvas(self.frame_container, width=c_w, height=c_h, bg="#D2B48C")
        self.canvas.pack(side=tk.LEFT)
        
        self.panel = tk.Frame(self.frame_container, width=250)
        self.panel.pack_propagate(False)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        self.btn_train = tk.Button(self.panel, text="開始自我對弈訓練", command=self.start_training)
        self.btn_train.pack(pady=10)
        
        self.btn_play = tk.Button(self.panel, text="與 AI 對戰 (人先手)", command=self.start_game_vs_ai)
        self.btn_play.pack(pady=10)
        
        self.lbl_status = tk.Label(self.panel, text="準備就緒", wraplength=180, font=("Arial", 12))
        self.lbl_status.pack(pady=20)

        self.turn_status = tk.Label(self.panel, text="", wraplength=150, font=("Arial", 12))
        self.turn_status.pack(pady=30)

        self.selected_pos = None
        self.human_playing = False
        self.training_running = False
        
        self.draw_board()

    def check_file(self):
        if not os.path.exists(self.cfg.SAVE_PATH):
            os.makedirs(self.cfg.SAVE_PATH) # 如果資料夾不存在就建立

    def draw_board(self):
        self.canvas.delete("all")
        w = self.cfg.CELL_SIZE
        h = self.cfg.CELL_SIZE
        
        # 畫線
        for i in range(self.cfg.BOARD_WIDTH + 1):
            self.canvas.create_line(i*w, 0, i*w, self.cfg.BOARD_HEIGHT * h)
        
        for i in range(self.cfg.BOARD_HEIGHT + 1):
            self.canvas.create_line(0, i*h, self.cfg.BOARD_WIDTH * w, i*h)
        
        
        for i in range(self.cfg.NUM_PIECES):
            r, c = divmod(i, self.cfg.BOARD_WIDTH)
            x = c * w + w//2
            y = r * h + h//2
            piece = self.env.board[i]
            
            if piece != self.cfg.EMPTY:
                color = "#555555" # Hidden color
                text = "?"
                text_color = "white"
                
                if piece != self.cfg.HIDDEN:
                    if piece in self.cfg.RED_PIECES:
                        color = "#FFCCCC"
                        text_color = "#CC0000"
                    else:
                        color = "#CCCCFF"
                        text_color = "#000000"
                    text = self.cfg.PIECE_NAMES[piece]
                
                outline = "black"
                width = 1
                if i == self.selected_pos:
                    outline = "yellow"
                    width = 3
                
                # Draw Piece
                self.canvas.create_oval(x-30, y-30, x+30, y+30, fill=color, outline=outline, width=width)
                self.canvas.create_text(x, y, text=text, fill=text_color, font=("Arial", 20, "bold"))
        
        # 顯示目前是誰的回合
        if not self.training_running:
            turn_text = "紅方回合" if self.env.turn == 0 else "黑方回合"
            self.turn_status.config(text=turn_text)

    def canvas_click(self, event):
        if not self.human_playing or self.training_running: 
            return
        if self.env.game_over: 
            return
        if self.env.turn != 0: 
            return # 假設人類永遠是 Player 0 (先手)

        w = self.cfg.CELL_SIZE
        h = self.cfg.CELL_SIZE
        c = event.x // w
        r = event.y // h
        pos = r * self.cfg.BOARD_WIDTH + c
        
        if c >= self.cfg.BOARD_WIDTH or r >= self.cfg.BOARD_HEIGHT: return

        # --- 點擊邏輯 (Source -> Destination) ---
        
        # 情況 1: 翻牌 (點擊蓋著的牌)
        if self.env.board[pos] == self.cfg.HIDDEN:
             # 翻牌動作: src == dst
             self.human_step(pos * self.cfg.NUM_PIECES + pos)
             return

        # 情況 2: 選取棋子 (Source)
        if self.selected_pos is None:
            # 只能選自己的棋子 (或尚未決定顏色時翻開的棋子)
            # 簡單檢查：如果是空的不能選
            if self.env.board[pos] != self.cfg.EMPTY:
                # 嚴格檢查是否為己方棋子
                # 注意：_is_my_piece 已經處理了 COLOR_UNKNOWN 的情況
                if self.env._is_my_piece(self.env.board[pos], 0):
                    self.selected_pos = pos
                    self.draw_board()
                    self.lbl_status.config(text="請選擇目標位置...")
                else:
                    self.lbl_status.config(text="這不是你的棋子")
        
        # 情況 3: 已選取棋子，點擊目標 (Destination)
        else:
            src = self.selected_pos
            dst = pos
            
            # 如果點擊自己，取消選取
            if src == dst:
                self.selected_pos = None
                self.draw_board()
                self.lbl_status.config(text="取消選取")
                return

            # 嘗試移動
            action = src * self.cfg.NUM_PIECES + dst
            self.human_step(action)

    def human_step(self, action):
        # 檢查合法性
        legal_mask = self.env.get_legal_actions(0)
        if not legal_mask[action]:
            self.lbl_status.config(text="無效移動！")
            self.selected_pos = None
            self.draw_board()
            return
            
        obs, reward, done, info = self.env.step(action)
        self.selected_pos = None
        self.draw_board()
        
        if done:
            self.show_winner()
            return
            
        # AI 回合
        self.lbl_status.config(text="AI 思考中...")
        self.root.after(500, self.ai_step)

    def ai_step(self):
        if self.env.game_over:
            return
        
        state, turn, eaten_state = self.env.get_state()
        mask = self.env.get_legal_actions(turn)
        action = self.agent.evaluate_action(state, turn, mask, eaten_state)        
        obs, reward, done, info = self.env.step(action)
        
        self.draw_board()
        self.lbl_status.config(text="輪到你了")
        
        if done:
            self.show_winner()

    def show_winner(self):
        winner_color = self.env.winner
        if winner_color == self.cfg.COLOR_RED: w_txt = "紅方"
        elif winner_color == self.cfg.COLOR_BLACK: w_txt = "黑方"
        else: w_txt = "和局"
        messagebox.showinfo("遊戲結束", f"結果: {w_txt} 獲勝")
        self.human_playing = False

    def start_game_vs_ai(self):
        self.training_running = False
        self.env.reset()
        self.human_playing = True
        self.selected_pos = None
        self.agent.load_model("C:\\workspace\\ChineseChess\\ChineseChess\\Save\\model_32.0.pt")
        self.draw_board()
        self.lbl_status.config(text="遊戲開始！請點選棋子")
        self.canvas.bind("<Button-1>", self.canvas_click)

    def start_training(self):
        """
            避免重複訓練
        """
        if self.training_running: 
            return
        self.training_running = True
        self.human_playing = False
        thread = threading.Thread(target=self.train_loop, daemon=True)
        thread.start()

    # =========================================================================
    # [新增] 輔助方法：將 turn 索引 (0/1) 轉換為顏色 (COLOR_RED/COLOR_BLACK)
    # =========================================================================
    def _get_player_color(self, turn_index):
        """
        根據 turn 索引取得該玩家的實際顏色。
        
        turn_index=0 的玩家的顏色 = self.env.my_color
        turn_index=1 的玩家的顏色 = 1 - self.env.my_color
        
        注意：必須在第一次翻牌後 (my_color 已確定) 才能呼叫。
        """
        if self.env.my_color == self.cfg.COLOR_UNKNOWN:
            # 尚未決定顏色 (不應該發生在需要判斷顏色的時候)
            # 安全起見給一個預設值
            return self.cfg.COLOR_RED if turn_index == 0 else self.cfg.COLOR_BLACK
        
        if turn_index == 0:
            return self.env.my_color
        else:
            return 1 - self.env.my_color

    # =========================================================================
    # [新增] 輔助方法：為指定顏色的記憶體添加獎勵
    # =========================================================================
    def _add_reward_to_memory(self, color, reward_value, terminal=False):
        """
            將額外獎勵加到指定顏色玩家的最後一步記憶中。
            用於遊戲結束時為非當前行動者分配勝負/和局獎勵。

            亦可用於需要額外懲罰的情況。
        """
        if color == self.cfg.COLOR_RED:
            if len(self.memory_red.rewards) > 0:
                self.memory_red.rewards[-1] += reward_value
                if terminal:
                    self.memory_red.is_terminals[-1] = True
        else:
            if len(self.memory_black.rewards) > 0:
                self.memory_black.rewards[-1] += reward_value
                if terminal:
                    self.memory_black.is_terminals[-1] = True

    # =========================================================================
    # 主訓練迴圈 (修正版)
    # =========================================================================
    def train_loop(self):
        for i_episode in range(1, self.cfg.MAX_EPISODES + 1):
            
            if not self.training_running: 
                break

            state, _, eaten_state = self.env.reset()
            step_count = 0
            # [修正] 追蹤單局獎勵 (不再使用 memory 累計值)
            ep_rewards = {'red': 0.0, 'black': 0.0}

            while True:
                current_turn = self.env.turn
                mask = self.env.get_legal_actions(current_turn)

                # --- 無合法動作：當前玩家被困住 → 輸了 ---
                if not any(mask):
                    winner_turn = 1 - current_turn
                    winner_color = self._get_player_color(winner_turn)
                    loser_color = self._get_player_color(current_turn)
                    
                    # 懲罰輸家 (修改其最後一步的獎勵)
                    self._add_reward_to_memory(loser_color, self.cfg.REWARD_LOSE, terminal=True)
                    # 獎勵贏家 (因為沒有經過 env.step，贏家不會從 step 拿到獎勵)
                    self._add_reward_to_memory(winner_color, self.cfg.REWARD_WIN, terminal=True)
                    
                    # 追蹤單局獎勵
                    loser_key = 'red' if loser_color == self.cfg.COLOR_RED else 'black'
                    winner_key = 'red' if winner_color == self.cfg.COLOR_RED else 'black'
                    ep_rewards[loser_key] += self.cfg.REWARD_LOSE
                    ep_rewards[winner_key] += self.cfg.REWARD_WIN
                    
                    self.env.game_over = True
                    self.env.winner = winner_color
                    break

                # 選擇動作
                action, log_prob = self.agent.select_action(state, current_turn, mask, eaten_state)
                
                # 執行動作
                next_state_info, reward, done, info = self.env.step(action)

                print(next_state_info[0])
                print("*"*120)

                next_state, next_turn, next_eaten_state = next_state_info
                step_count += 1

                # =============================================================
                # [修正 Bug 7] 按顏色儲存到正確的記憶體
                # 原本用 current_turn (0/1) 直接對應 memory_red/black，
                # 但 turn=0 不一定是紅方！必須透過 my_color 映射。
                # =============================================================
                current_color = self._get_player_color(current_turn)

                
                
                color_key = 'red' if current_color == self.cfg.COLOR_RED else 'black'
                ep_rewards[color_key] += reward  # 追蹤單局獎勵
                
                if current_color == self.cfg.COLOR_RED:
                    self.memory_red.states.append(torch.FloatTensor(state))
                    self.memory_red.eaten_states.append(torch.FloatTensor(eaten_state))
                    self.memory_red.turns.append(current_turn)
                    self.memory_red.masks.append(torch.BoolTensor(mask))
                    self.memory_red.actions.append(torch.tensor(action))
                    self.memory_red.logprobs.append(torch.tensor(log_prob))
                    self.memory_red.rewards.append(reward)
                    self.memory_red.is_terminals.append(done)
                else:
                    self.memory_black.states.append(torch.FloatTensor(state))
                    self.memory_black.eaten_states.append(torch.FloatTensor(eaten_state))
                    self.memory_black.turns.append(current_turn)
                    self.memory_black.masks.append(torch.BoolTensor(mask))
                    self.memory_black.actions.append(torch.tensor(action))
                    self.memory_black.logprobs.append(torch.tensor(log_prob))
                    self.memory_black.rewards.append(reward)
                    self.memory_black.is_terminals.append(done)

                state = next_state
                eaten_state = next_eaten_state


                if info.get("Eaten_reward", 0 ) != 0:
                    opponent_color = self._get_player_color(1 - current_turn)
                    opponent_key = 'red' if opponent_color == self.cfg.COLOR_RED else 'black'
                    self._add_reward_to_memory(self._get_player_color(1 - current_turn), info.get("Eaten_reward", 0 ), terminal=False)
                    ep_rewards[opponent_key] += self.cfg.REWARD_EATEN


                
                # --- 遊戲正常結束 (吃光棋子、步數上限、重複局面) ---
                if done:
                    if self.env.winner is not None:
                        # =================================================
                        # [修正] 正確分配勝負獎勵
                        # env.step() 已經給當前行動者獎勵 (WIN 或 LOSE)
                        # 這裡只需要處理「對手」(非當前行動者) 的獎勵
                        # =================================================
                        winner_color = self.env.winner
                        current_color = self._get_player_color(current_turn)
                        opponent_color = self._get_player_color(1 - current_turn)
                        
                        opponent_key = 'red' if opponent_color == self.cfg.COLOR_RED else 'black'
                        if current_color == winner_color:
                            # 當前行動者贏了 (已從 step 拿到 REWARD_WIN)
                            # → 對手輸了，需要加 LOSE 到對手的最後一步
                            self._add_reward_to_memory(opponent_color, self.cfg.REWARD_LOSE, terminal=True)
                            ep_rewards[opponent_key] += self.cfg.REWARD_LOSE
                        else:
                            # 當前行動者輸了 (已從 step 拿到 REWARD_LOSE)
                            # → 對手贏了，需要加 WIN 到對手的最後一步
                            self._add_reward_to_memory(opponent_color, self.cfg.REWARD_WIN, terminal=True)
                            ep_rewards[opponent_key] += self.cfg.REWARD_WIN
                    else:
                        # 和局：當前行動者已從 step 拿到 REWARD_DRAW
                        # → 對手也需要 REWARD_DRAW
                        opponent_color = self._get_player_color(1 - current_turn)
                        self._add_reward_to_memory(opponent_color, self.cfg.REWARD_DRAW, terminal=True)
                        opponent_key = 'red' if opponent_color == self.cfg.COLOR_RED else 'black'
                        ep_rewards[opponent_key] += self.cfg.REWARD_DRAW
                    
                    break

            # --- 計算本局獎勵 (單局追蹤) ---
            episodic_red_reward = ep_rewards['red']
            episodic_black_reward = ep_rewards['black']
            episodic_reward = episodic_red_reward + episodic_black_reward

            # --- 記錄到診斷工具 ---
            self.diagnostics.log_episode(
                i_episode, episodic_reward, episodic_red_reward, episodic_black_reward,
                step_count, self.env.winner, self.env.my_color
            )
            
            # --- Update Model ---
            if i_episode % self.cfg.UPDATE_FREQ == 0:
                diag_red = {}
                diag_black = {}
                
                # [修正 Bug 1] 紅方與黑方都要更新
                if len(self.memory_red.states) > 0:
                    diag_red = self.agent.update(self.memory_red)
                
                if len(self.memory_black.states) > 0:
                    diag_black = self.agent.update(self.memory_black)
                
                # 合併診斷指標 (取平均)
                combined_diag = {}
                diag_list = [d for d in [diag_red, diag_black] if d]
                if diag_list:
                    all_keys = diag_list[0].keys()
                    for k in all_keys:
                        combined_diag[k] = sum(d.get(k, 0) for d in diag_list) / len(diag_list)
                    self.diagnostics.log_update(i_episode, combined_diag)
                
                # 清空記憶
                self.memory_red.clear_memory()
                self.memory_black.clear_memory()
            
            # --- 定期印出診斷摘要 + 更新 UI ---
            if i_episode % self.cfg.PRINT_FREQ == 0:
                self.diagnostics.print_summary(i_episode)
                
                # 使用 after 在主執行緒更新 UI
                self.root.after(0, lambda ep=i_episode, r=episodic_reward: self.update_train_ui(ep, r))
                # 稍微畫一下最後的盤面讓使用者看到有在動
                self.root.after(0, self.draw_board)
                time.sleep(0.1) # 稍微暫停避免 UI 卡死


            if i_episode % self.cfg.CHECKPOINT_IDNEX == 0:
                self.agent.save_model(os.path.join(self.cfg.SAVE_PATH, f"model_{i_episode / self.cfg.CHECKPOINT_IDNEX}.pt"))

        self.training_running = False
        self.root.after(0, lambda: self.lbl_status.config(text="訓練完成！"))

    def update_train_ui(self, ep, r):
        self.lbl_status.config(text=f"Episode: {ep}, Reward: {r:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkChessGUI(root)
    root.mainloop()
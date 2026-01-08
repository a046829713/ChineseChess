import tkinter as tk
from tkinter import messagebox
import threading
import time
import copy
from config import GameConfig
from env import DarkChessEnv
from model import PPOAgent
from model import Memory
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

        self.memory_red = Memory()   # 專存紅方 (Turn 0) 的 (s, a, r)
        self.memory_black = Memory() # 專存黑方 (Turn 1) 的 (s, a, r)
        
        self.frame_container = tk.Frame(root)
        self.frame_container.pack()

        c_w = self.cfg.BOARD_WIDTH * self.cfg.CELL_SIZE
        c_h = self.cfg.BOARD_HEIGHT * self.cfg.CELL_SIZE
        self.canvas = tk.Canvas(self.frame_container, width=c_w, height=c_h, bg="#D2B48C")
        self.canvas.pack(side=tk.LEFT)
        
        self.panel = tk.Frame(self.frame_container)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        self.btn_train = tk.Button(self.panel, text="開始自我對弈訓練", command=self.start_training)
        self.btn_train.pack(pady=10)
        
        self.btn_play = tk.Button(self.panel, text="與 AI 對戰 (人先手)", command=self.start_game_vs_ai)
        self.btn_play.pack(pady=10)
        
        self.lbl_status = tk.Label(self.panel, text="準備就緒", wraplength=150, font=("Arial", 12))
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
        
        state, turn = self.env.get_state()
        mask = self.env.get_legal_actions(turn)
        action = self.agent.evaluate_action(state, turn, mask)        
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
        self.agent.load_model("C:\workspace\chinesechess\Save\model_22.0.pt")
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

    def train_loop(self):
        for i_episode in range(1, self.cfg.MAX_EPISODES + 1):
            
            if not self.training_running: 
                break
            


            state_obs, _ = self.env.reset()
            state = state_obs
            episodic_reward = 0
            episodic_red_reward = 0
            episodic_black_reward = 0

            last_mask_red = None
            last_mask_black = None

            while True:
                current_turn = self.env.turn 
                mask = self.env.get_legal_actions(current_turn)

                if not any(mask): # 如果沒有任何 True
                    # 當前玩家被困住 -> 輸了 (LOSE)
                    # 對手 -> 贏了 (WIN)
                    winner = 1 - current_turn                    
                    self.assign_loser_punishment(winner)

                    
                    # 結束這局
                    break

                # [修正] 訓練時傳入 Turn
                action, log_prob = self.agent.select_action(state, current_turn, mask)
                
                next_state_info, reward, done, _ = self.env.step(action)
                next_state, next_turn = next_state_info
                
                # 4. [雙記憶體儲存邏輯]
                # 我們將資料存入「當前行動者」的記憶體中
                # 注意：這時候的 reward 通常只是「吃子獎勵」，「勝負獎勵」可能還沒出來
                
                if current_turn == 0: # 紅方
                    self.memory_red.states.append(torch.FloatTensor(state))
                    self.memory_red.turns.append(current_turn)
                    self.memory_red.masks.append(torch.BoolTensor(mask))
                    self.memory_red.actions.append(torch.tensor(action))
                    self.memory_red.logprobs.append(torch.tensor(log_prob))
                    self.memory_red.rewards.append(reward) # 暫存當下獎勵
                    self.memory_red.is_terminals.append(done)
                    episodic_red_reward += reward
                else: # 黑方
                    self.memory_black.states.append(torch.FloatTensor(state))
                    self.memory_black.turns.append(current_turn)
                    self.memory_black.masks.append(torch.BoolTensor(mask))
                    self.memory_black.actions.append(torch.tensor(action))
                    self.memory_black.logprobs.append(torch.tensor(log_prob))
                    self.memory_black.rewards.append(reward) # 暫存當下獎勵
                    self.memory_black.is_terminals.append(done)
                    episodic_black_reward +=reward

                episodic_reward += reward
                state = next_state
                
                # 5. [處理正常結束] (例如清空棋子或步數上限)
                if done:
                    print("self.env.winner:", self.env.winner,"最後獎勵為:",reward)
                    if self.env.winner is not None:
                        self.assign_loser_punishment(self.env.winner)
                    else:
                        self.assign_draw_punishment(current_turn)

                    if self.env.winner == self.cfg.COLOR_RED:
                        episodic_black_reward += self.cfg.REWARD_LOSE

                    elif self.env.winner == self.cfg.COLOR_BLACK:
                        episodic_red_reward += self.cfg.REWARD_LOSE
                    
                    break
                
                    
                    
                
            print("目前集數:",i_episode)
            print("紅方獎勵:",episodic_red_reward)
            print("黑方獎勵:",episodic_black_reward)
            print("*"*120)
            
            # Update Model
            if i_episode % self.cfg.UPDATE_FREQ == 0:
                # 你可以選擇把兩個 memory 合併，或者分開 update
                # 這裡建議分開呼叫，邏輯比較簡單清晰
                
                # 更新紅方經驗 (學習如何贏)
                if len(self.memory_red.states) > 0:
                    self.agent.update(self.memory_red)
                
                # 更新黑方經驗 (學習如何不輸)
                if len(self.memory_black.states) > 0:
                    self.agent.update(self.memory_black)
                
                # 清空記憶
                self.memory_red.clear_memory()
                self.memory_black.clear_memory()
            
            # Update UI
            if i_episode % self.cfg.PRINT_FREQ == 0:
                # 使用 after 在主執行緒更新 UI
                self.root.after(0, lambda ep=i_episode, r=episodic_reward: self.update_train_ui(ep, r))
                # 稍微畫一下最後的盤面讓使用者看到有在動
                self.root.after(0, self.draw_board)
                time.sleep(0.1) # 稍微暫停避免 UI 卡死


            if i_episode % self.cfg.CHECKPOINT_IDNEX == 0:
                self.agent.save_model(self.cfg.SAVE_PATH + f"./model_{i_episode /self.cfg.CHECKPOINT_IDNEX}.pt")

        self.training_running = False
        self.root.after(0, lambda: self.lbl_status.config(text="訓練完成！"))

    def update_train_ui(self, ep, r):
        self.lbl_status.config(text=f"Episode: {ep}, Reward: {r:.2f}")
    
    def assign_draw_punishment(self, current_turn):
        """
        當發生和局時，除了當前行動者(已在step獲得懲罰)，也要懲罰上一手(等待中)的玩家。
        """
        waiting_player = 1 - current_turn
        draw_penalty = self.cfg.REWARD_DRAW
        
        if waiting_player == self.cfg.COLOR_RED:
            if len(self.memory_red.rewards) > 0:
                self.memory_red.rewards[-1] += draw_penalty
                self.memory_red.is_terminals[-1] = True
        else:
            if len(self.memory_black.rewards) > 0:
                self.memory_black.rewards[-1] += draw_penalty
                self.memory_black.is_terminals[-1] = True
    
    def assign_loser_punishment(self, winner_color):
        """
        只負責懲罰輸家。
        因為贏家的獎勵在 env.step() 裡已經拿到了 (REWARD_WIN)。
        """
        lose_reward = self.cfg.REWARD_LOSE # 例如 -10
        
        # 如果紅方贏了，黑方(上一手)需要被懲罰
        if winner_color == self.cfg.COLOR_RED:
            if len(self.memory_black.rewards) > 0:
                self.memory_black.rewards[-1] += lose_reward
                self.memory_black.is_terminals[-1] = True
        
        # 如果黑方贏了，紅方(上一手)需要被懲罰
        elif winner_color == self.cfg.COLOR_BLACK:
            if len(self.memory_red.rewards) > 0:
                self.memory_red.rewards[-1] += lose_reward
                self.memory_red.is_terminals[-1] = True

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkChessGUI(root)
    root.mainloop()
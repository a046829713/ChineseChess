class GameConfig:
    def __init__(self):
        # --- 棋盤設定 ---
        self.BOARD_WIDTH = 8
        self.BOARD_HEIGHT = 4
        self.NUM_PIECES = 32
        
        # --- 動作空間 ---
        # 總動作空間：32個起點 * 32個終點 = 1024
        self.TOTAL_ACTIONS = 1024 

        # --- PPO 超參數 (Hyperparameters) ---
        self.LEARNING_RATE = 0.0001
        self.GAMMA = 0.95       # 折扣因子 (降低以加速終局信號傳遞)
        self.GAE_LAMBDA = 0.95
        self.EPS_CLIP = 0.2     # PPO Clip 範圍
        self.K_EPOCHS = 3      # 最多迭代次數 (配合 KL 早停)
        self.KL_TARGET = 0.015  # KL divergence 目標值 (超過就停止迭代)
        
        # --- 神經網路架構 ---
        self.EMBED_DIM = 32     # 棋子 Embedding 維度
        self.HIDDEN_DIM_1 = 256 # 第一層隱藏層
        self.HIDDEN_DIM_2 = 128 # 第二層隱藏層

        # --- 獎勵機制 (Rewards) ---
        self.REWARD_FLIP = 0.02      # 翻牌獎勵 (提高以鼓勵探索新棋子)
        self.REWARD_WIN = 3.0       # 原本 30.0
        self.REWARD_LOSE = -3.0     # 原本 -10.0
        
        
        
        self.REWARD_DRAW = -0.1     # 原本 -8.0
        self.REWARD_LOSS_DRAW = -0.5
        self.REWARD_MOVE = -0.005   # 原本 -0.05
        
        self.PIECE_VALUES = {
            1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3, 6: 0.5, 7: 0.2,  # 紅方
            8: 0.7, 9: 0.6, 10: 0.5, 11: 0.4, 12: 0.3, 13: 0.5, 14: 0.2, # 黑方
        }

        
        self.REWARD_INVALID = -0.1

        # --- 訓練設定 ---
        self.MAX_EPISODES = 10000    # 總訓練局數

        self.UPDATE_FREQ = 10       # 每幾局更新一次網路
        self.PRINT_FREQ = 50        # 每幾局印出一次 Log

        # 棋子編碼
        self.EMPTY = 0
        self.HIDDEN = -1

        # 紅方: 1-7 (帥, 仕, 相, 俥, 傌, 炮, 兵)
        # 黑方: 8-14 (將, 士, 象, 車, 馬, 包, 卒)
        self.RED_PIECES = [1, 2, 3, 4, 5, 6, 7]
        self.BLACK_PIECES = [8, 9, 10, 11, 12, 13, 14]

        # 顏色標記
        self.COLOR_RED = 0
        self.COLOR_BLACK = 1
        self.COLOR_UNKNOWN = -1

        self.PIECE_NAMES = {
            1: '帥', 2: '仕', 3: '相', 4: '俥', 5: '傌', 6: '炮', 7: '兵',
            8: '將', 9: '士', 10: '象', 11: '車', 12: '馬', 13: '包', 14: '卒',
            self.HIDDEN: '0', self.EMPTY: ''
        }

        self.CELL_SIZE = 90
        self.SAVE_PATH ='Save'
        self.CHECKPOINT_IDNEX = 100
        
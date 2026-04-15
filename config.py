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
        self.LEARNING_RATE = 0.0005
        self.GAMMA = 0.99       # 折扣因子
        self.EPS_CLIP = 0.2     # PPO Clip 範圍
        self.K_EPOCHS = 4       # 每次更新的訓練次數
        
        # --- 神經網路架構 ---
        self.EMBED_DIM = 32     # 棋子 Embedding 維度
        self.HIDDEN_DIM_1 = 256 # 第一層隱藏層
        self.HIDDEN_DIM_2 = 128 # 第二層隱藏層

        # --- 獎勵機制 (Rewards) ---
        self.REWARD_FLIP = 0.002     # 翻牌獎勵 (鼓勵開局)
        self.REWARD_MOVE = -0.01     # 普通移動
        self.REWARD_EAT = 1.0        # 吃子獎勵 (提高以鼓勵進攻)
        self.REWARD_WIN = 10.0      # 獲勝獎勵
        self.REWARD_LOSE = -10.0    # 落敗懲罰
        self.REWARD_DRAW = -5.0      # 和棋 (提高懲罰，避免 AI 故意拖延)
        
        
    

        
        self.REWARD_INVALID = -0.1

        # --- 訓練設定 ---
        self.MAX_EPISODES = 10000    # 總訓練局數

        self.UPDATE_FREQ = 1       # 每幾局更新一次網路
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
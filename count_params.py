from model import PPOAgent
from config import GameConfig

cfg = GameConfig()
agent = PPOAgent(cfg)

total = 0
print("Layer                                    | Params")
print("-" * 60)
for name, p in agent.policy.named_parameters():
    n = p.numel()
    total += n
    print(f"{name:40s} | {n:>10,}")
print("-" * 60)
print(f"TOTAL                                    | {total:>10,}")
print()
print("=== Architecture Summary ===")
print(f"Embedding: 16 tokens -> 64 dim")
print(f"Conv backbone: 64 channels, 3 ResBlocks (kernel=3x3)")
print(f"Board: 4x8 = 32 positions")
print(f"CNN output: 64 * 4 * 8 = 2048")
print(f"FC shared: 2080 -> 512")
print(f"Actor head: 512 -> 256 -> 1024")
print(f"Value head: 512 -> 128 -> 1")
print(f"Total params: {total:,}")

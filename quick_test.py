import torch
from config import GameConfig
from env import DarkChessEnv
from model import PPOAgent, Memory

cfg = GameConfig()
cfg.MAX_EPISODES = 20
cfg.UPDATE_FREQ = 5
env = DarkChessEnv(cfg)
agent = PPOAgent(cfg)
memory_red = Memory()
memory_black = Memory()

def get_color(env, cfg, t):
    if env.my_color == cfg.COLOR_UNKNOWN:
        return cfg.COLOR_RED if t == 0 else cfg.COLOR_BLACK
    return env.my_color if t == 0 else 1 - env.my_color

for ep in range(1, 21):
    state, _, eaten = env.reset()
    steps = 0
    while True:
        turn = env.turn
        mask = env.get_legal_actions(turn)
        if not any(mask):
            env.game_over = True
            break
        action, lp = agent.select_action(state, turn, mask, eaten)
        ns, reward, done, _ = env.step(action)
        state, _, eaten = ns
        steps += 1
        color = get_color(env, cfg, turn)
        mem = memory_red if color == cfg.COLOR_RED else memory_black
        mem.states.append(torch.FloatTensor(state))
        mem.eaten_states.append(torch.FloatTensor(eaten))
        mem.turns.append(turn)
        mem.masks.append(torch.BoolTensor(mask))
        mem.actions.append(torch.tensor(action))
        mem.logprobs.append(torch.tensor(lp))
        mem.rewards.append(reward)
        mem.is_terminals.append(done)
        if done:
            break

    if ep % cfg.UPDATE_FREQ == 0:
        d1 = agent.update(memory_red) if memory_red.states else {}
        d2 = agent.update(memory_black) if memory_black.states else {}
        memory_red.clear_memory()
        memory_black.clear_memory()
        for label, d in [("RED", d1), ("BLACK", d2)]:
            if d:
                ent = d["entropy"]
                kl = d["kl_divergence"]
                cl = d["critic_loss"]
                gn = d["grad_norm"]
                rm = d["ratio_mean"]
                print(f"Ep {ep} [{label}] entropy={ent:.3f} kl={kl:.4f} critic={cl:.3f} grad={gn:.2f} ratio={rm:.3f}")

print("\n=== 20 episodes completed! ===")

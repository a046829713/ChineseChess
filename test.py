returns = []

rewards = [1,2,3]
gamma = 0.95

for t in range(len(rewards)):
    G_t = 0
    # 每個時間步都要往未來遍歷到底
    for k in range(t, len(rewards)):
        G_t += rewards[k] * (gamma ** (k - t))
    returns.append(G_t)

print(returns)
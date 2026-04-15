"""
訓練診斷工具 (Training Diagnostics)
用於即時監控 PPO 暗棋 RL 訓練過程中的各項指標，幫助排查收斂問題。

功能：
1. Episode 級別指標：獎勵、勝負率、局長度
2. Update 級別指標：Loss 分解、梯度、KL divergence、Clip fraction
3. 自動收斂問題偵測與警告
4. CSV 輸出供事後分析
"""

import csv
import os
import time
import numpy as np
from collections import deque


class TrainingDiagnostics:
    def __init__(self, save_dir='Save', window_size=100):
        self.save_dir = save_dir
        self.window_size = window_size

        # --- 滾動統計 (Rolling Metrics) ---
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_red_rewards = deque(maxlen=window_size)
        self.episode_black_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.results = deque(maxlen=window_size)  # 'red_win', 'black_win', 'draw'

        # --- Update 指標歷史 ---
        self.update_history = deque(maxlen=window_size)

        # --- 全域計數器 ---
        self.total_red_wins = 0
        self.total_black_wins = 0
        self.total_draws = 0
        self.total_episodes = 0

        # --- CSV 檔案路徑 ---
        self.episode_csv = os.path.join(save_dir, 'episode_log.csv')
        self.update_csv = os.path.join(save_dir, 'update_log.csv')
        self._init_csv()

        # --- 計時 ---
        self.start_time = time.time()

    def _init_csv(self):
        """初始化 CSV 檔案 (寫入表頭)"""
        os.makedirs(self.save_dir, exist_ok=True)

        with open(self.episode_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'red_reward', 'black_reward',
                'steps', 'result', 'elapsed_time'
            ])

        with open(self.update_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'actor_loss', 'critic_loss', 'entropy', 'total_loss',
                'clip_fraction', 'kl_divergence', 'grad_norm',
                'value_pred_mean', 'advantage_mean', 'ratio_mean'
            ])

    def log_episode(self, episode, total_reward, red_reward, black_reward,
                    steps, winner_color, my_color):
        """記錄一局的結果"""
        self.total_episodes += 1
        self.episode_rewards.append(total_reward)
        self.episode_red_rewards.append(red_reward)
        self.episode_black_rewards.append(black_reward)
        self.episode_lengths.append(steps)

        # 判定結果
        if winner_color == 0:  # COLOR_RED
            result = 'red_win'
            self.total_red_wins += 1
        elif winner_color == 1:  # COLOR_BLACK
            result = 'black_win'
            self.total_black_wins += 1
        else:
            result = 'draw'
            self.total_draws += 1

        self.results.append(result)

        # 寫入 CSV
        elapsed = time.time() - self.start_time
        with open(self.episode_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, f'{total_reward:.4f}', f'{red_reward:.4f}', f'{black_reward:.4f}',
                steps, result, f'{elapsed:.1f}'
            ])

    def log_update(self, episode, diagnostics):
        """記錄一次 PPO 更新的指標"""
        if not diagnostics:
            return

        self.update_history.append(diagnostics)

        with open(self.update_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                f'{diagnostics.get("actor_loss", 0):.6f}',
                f'{diagnostics.get("critic_loss", 0):.6f}',
                f'{diagnostics.get("entropy", 0):.6f}',
                f'{diagnostics.get("total_loss", 0):.6f}',
                f'{diagnostics.get("clip_fraction", 0):.4f}',
                f'{diagnostics.get("kl_divergence", 0):.6f}',
                f'{diagnostics.get("grad_norm", 0):.4f}',
                f'{diagnostics.get("value_pred_mean", 0):.6f}',
                f'{diagnostics.get("advantage_mean", 0):.6f}',
                f'{diagnostics.get("ratio_mean", 0):.6f}',
            ])

    def print_summary(self, episode):
        """印出訓練摘要"""
        if len(self.episode_rewards) == 0:
            return

        elapsed = time.time() - self.start_time
        eps_per_sec = self.total_episodes / elapsed if elapsed > 0 else 0

        # --- 獎勵統計 ---
        avg_reward = np.mean(self.episode_rewards)
        avg_red = np.mean(self.episode_red_rewards)
        avg_black = np.mean(self.episode_black_rewards)
        avg_length = np.mean(self.episode_lengths)

        # --- 勝率統計 ---
        recent_results = list(self.results)
        n = len(recent_results)
        red_rate = recent_results.count('red_win') / n * 100
        black_rate = recent_results.count('black_win') / n * 100
        draw_rate = recent_results.count('draw') / n * 100

        print(f"\n{'=' * 80}")
        print(f"  [Episode {episode}] | Elapsed: {elapsed:.0f}s | Speed: {eps_per_sec:.1f} ep/s")
        print(f"{'=' * 80}")
        print(f"  Avg Reward (last {n} ep): {avg_reward:>10.4f}")
        print(f"  Avg Red Reward:           {avg_red:>10.4f}")
        print(f"  Avg Black Reward:         {avg_black:>10.4f}")
        print(f"  Avg Episode Length:        {avg_length:>9.1f}")
        print(f"  Win Rate  - Red: {red_rate:>5.1f}% | Black: {black_rate:>5.1f}% | Draw: {draw_rate:>5.1f}%")
        print(f"  Cumulative- Red: {self.total_red_wins:>5d}  | Black: {self.total_black_wins:>5d}  | Draw: {self.total_draws:>5d}")

        # --- 最近一次更新的指標 ---
        if self.update_history:
            latest = self.update_history[-1]
            print(f"  {'─' * 40}")
            print(f"  Latest PPO Update Metrics:")
            print(f"    Actor Loss:     {latest.get('actor_loss', 0):>12.6f}")
            print(f"    Critic Loss:    {latest.get('critic_loss', 0):>12.6f}")
            print(f"    Entropy:        {latest.get('entropy', 0):>12.6f}")
            print(f"    Total Loss:     {latest.get('total_loss', 0):>12.6f}")
            print(f"    Clip Fraction:  {latest.get('clip_fraction', 0):>12.4f}")
            print(f"    KL Divergence:  {latest.get('kl_divergence', 0):>12.6f}")
            print(f"    Grad Norm:      {latest.get('grad_norm', 0):>12.4f}")
            print(f"    Value Mean:     {latest.get('value_pred_mean', 0):>12.6f}")
            print(f"    Advantage Mean: {latest.get('advantage_mean', 0):>12.6f}")
            print(f"    Ratio Mean:     {latest.get('ratio_mean', 0):>12.6f}")

            # 自動偵測問題
            self._check_warnings(latest)

        print(f"{'=' * 80}\n")

    def _check_warnings(self, metrics):
        """自動偵測潛在的收斂問題"""
        warnings = []

        # 1. Entropy 坍縮 (Policy Collapse)
        entropy = metrics.get('entropy', 999)
        if entropy < 0.1:
            warnings.append(
                f"⚠️  ENTROPY 極低 ({entropy:.4f}) — 策略可能已坍縮，失去探索能力！"
                f"\n       建議：增大 entropy 係數或降低學習率"
            )
        elif entropy < 0.5:
            warnings.append(
                f"⚠️  ENTROPY 偏低 ({entropy:.4f}) — 探索不足，策略趨向確定性"
            )

        # 2. KL Divergence 過大 (更新步幅過大)
        kl = abs(metrics.get('kl_divergence', 0))
        if kl > 0.1:
            warnings.append(
                f"⚠️  KL DIVERGENCE 過高 ({kl:.4f}) — 更新幅度過大，訓練可能不穩！"
                f"\n       建議：降低學習率或增大 K_EPOCHS"
            )

        # 3. 梯度異常
        grad_norm = metrics.get('grad_norm', 0)
        if grad_norm > 10:
            warnings.append(
                f"⚠️  GRADIENT NORM 過大 ({grad_norm:.2f}) — 可能發生梯度爆炸！"
                f"\n       建議：降低 clip_grad_norm 的 max_norm"
            )
        elif grad_norm < 1e-7:
            warnings.append(
                f"⚠️  GRADIENT NORM 趨近零 ({grad_norm:.2e}) — 可能發生梯度消失！"
                f"\n       建議：檢查網路架構或獎勵範圍"
            )

        # 4. Clip Fraction 過高
        clip_frac = metrics.get('clip_fraction', 0)
        if clip_frac > 0.5:
            warnings.append(
                f"⚠️  CLIP FRACTION 過高 ({clip_frac:.2f}) — 太多動作被裁剪，學習效率低！"
                f"\n       建議：降低學習率"
            )

        # 5. Ratio 偏離 1.0
        ratio = metrics.get('ratio_mean', 1.0)
        if abs(ratio - 1.0) > 0.5:
            warnings.append(
                f"⚠️  RATIO MEAN 偏離 1.0 ({ratio:.4f}) — 新舊策略差異過大！"
            )

        # 6. Critic Loss 過大
        critic_loss = metrics.get('critic_loss', 0)
        if critic_loss > 100:
            warnings.append(
                f"⚠️  CRITIC LOSS 過大 ({critic_loss:.2f}) — Value Function 預測值偏差極大！"
            )

        # 7. Advantage Mean 偏離 0 (正常化後應接近 0)
        adv_mean = abs(metrics.get('advantage_mean', 0))
        if adv_mean > 2.0:
            warnings.append(
                f"⚠️  ADVANTAGE MEAN 偏大 ({adv_mean:.4f}) — 基線估計可能不準確"
            )

        if warnings:
            print(f"  {'─' * 40}")
            print(f"  🔍 自動偵測警告:")
            for w in warnings:
                print(f"    {w}")

    def print_weight_stats(self, weight_stats):
        """印出模型權重統計 (from agent.get_weight_stats())"""
        if not weight_stats:
            return

        print(f"\n  {'─' * 40}")
        print(f"  Model Weight Statistics:")
        print(f"  {'Layer':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Norm':>10}")
        print(f"  {'─' * 95}")

        for name, stats in weight_stats.items():
            # 只顯示權重 (跳過 bias 以減少輸出量)
            if 'weight' in name:
                print(f"  {name:<45} "
                      f"{stats['mean']:>10.6f} "
                      f"{stats['std']:>10.6f} "
                      f"{stats['min']:>10.6f} "
                      f"{stats['max']:>10.6f} "
                      f"{stats['norm']:>10.4f}")


# --- 獨立執行：分析已儲存的 CSV 日誌 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='分析訓練日誌')
    parser.add_argument('--save_dir', type=str, default='Save', help='日誌目錄路徑')
    parser.add_argument('--last_n', type=int, default=100, help='分析最後 N 筆資料')
    args = parser.parse_args()

    episode_csv = os.path.join(args.save_dir, 'episode_log.csv')
    update_csv = os.path.join(args.save_dir, 'update_log.csv')

    # --- 分析 Episode 日誌 ---
    if os.path.exists(episode_csv):
        print(f"\n{'=' * 80}")
        print(f"  Episode Log Analysis (last {args.last_n} episodes)")
        print(f"{'=' * 80}")

        rows = []
        with open(episode_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            recent = rows[-args.last_n:]
            total_rewards = [float(r['total_reward']) for r in recent]
            red_rewards = [float(r['red_reward']) for r in recent]
            black_rewards = [float(r['black_reward']) for r in recent]
            steps = [int(r['steps']) for r in recent]
            results = [r['result'] for r in recent]

            n = len(recent)
            print(f"  Total Episodes:    {len(rows)}")
            print(f"  Analyzing last:    {n}")
            print(f"  Avg Total Reward:  {np.mean(total_rewards):.4f} (±{np.std(total_rewards):.4f})")
            print(f"  Avg Red Reward:    {np.mean(red_rewards):.4f} (±{np.std(red_rewards):.4f})")
            print(f"  Avg Black Reward:  {np.mean(black_rewards):.4f} (±{np.std(black_rewards):.4f})")
            print(f"  Avg Steps:         {np.mean(steps):.1f} (±{np.std(steps):.1f})")
            print(f"  Red Win Rate:      {results.count('red_win') / n * 100:.1f}%")
            print(f"  Black Win Rate:    {results.count('black_win') / n * 100:.1f}%")
            print(f"  Draw Rate:         {results.count('draw') / n * 100:.1f}%")

            # 趨勢分析：前半 vs 後半
            if n >= 20:
                half = n // 2
                first_half = total_rewards[:half]
                second_half = total_rewards[half:]
                trend = np.mean(second_half) - np.mean(first_half)
                trend_str = "↑ 上升" if trend > 0 else "↓ 下降" if trend < 0 else "→ 持平"
                print(f"\n  Reward Trend:      {trend_str} ({trend:+.4f})")
                print(f"    First half avg:  {np.mean(first_half):.4f}")
                print(f"    Second half avg: {np.mean(second_half):.4f}")
        else:
            print("  No episode data found.")
    else:
        print(f"  Episode log not found: {episode_csv}")

    # --- 分析 Update 日誌 ---
    if os.path.exists(update_csv):
        print(f"\n{'=' * 80}")
        print(f"  Update Log Analysis (last {args.last_n} updates)")
        print(f"{'=' * 80}")

        rows = []
        with open(update_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            recent = rows[-args.last_n:]
            n = len(recent)

            metrics = {
                'actor_loss': [], 'critic_loss': [], 'entropy': [],
                'total_loss': [], 'clip_fraction': [], 'kl_divergence': [],
                'grad_norm': [], 'value_pred_mean': [], 'ratio_mean': []
            }

            for r in recent:
                for key in metrics:
                    try:
                        metrics[key].append(float(r[key]))
                    except (ValueError, KeyError):
                        pass

            print(f"  Total Updates: {len(rows)}")
            print(f"  Analyzing last: {n}")
            print(f"\n  {'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {'─' * 68}")

            for key, values in metrics.items():
                if values:
                    print(f"  {key:<20} {np.mean(values):>12.6f} {np.std(values):>12.6f} "
                          f"{np.min(values):>12.6f} {np.max(values):>12.6f}")

            # 趨勢分析
            if len(metrics['total_loss']) >= 20:
                half = len(metrics['total_loss']) // 2
                first = np.mean(metrics['total_loss'][:half])
                second = np.mean(metrics['total_loss'][half:])
                trend = second - first
                trend_str = "↑ 上升" if trend > 0.001 else "↓ 下降" if trend < -0.001 else "→ 持平"
                print(f"\n  Loss Trend: {trend_str} ({trend:+.6f})")

            # Entropy 趨勢
            if len(metrics['entropy']) >= 20:
                half = len(metrics['entropy']) // 2
                first = np.mean(metrics['entropy'][:half])
                second = np.mean(metrics['entropy'][half:])
                trend = second - first
                if trend < -0.5:
                    print(f"  ⚠️  Entropy 持續下降 ({trend:+.4f}) — 策略可能坍縮")
                elif trend > 0.5:
                    print(f"  ✅ Entropy 上升 ({trend:+.4f}) — 探索增加")
        else:
            print("  No update data found.")
    else:
        print(f"  Update log not found: {update_csv}")

    print(f"\n{'=' * 80}")
    print(f"  Analysis complete.")
    print(f"{'=' * 80}\n")

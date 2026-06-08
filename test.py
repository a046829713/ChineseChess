# ... 前面初始化省略 ...
            step_count = 0
            ep_rewards = {'red': 0.0, 'black': 0.0}

            while True:
                current_turn = self.env.turn
                mask = self.env.get_legal_actions(current_turn)

                # --- 無合法動作：當前玩家被困住 → 輸了 ---
                if not any(mask):
                    # (這部分你的原本代碼寫得很好，保持不變)
                    winner_turn = 1 - current_turn
                    winner_color = self._get_player_color(winner_turn)
                    loser_color = self._get_player_color(current_turn)
                    
                    self._add_reward_to_memory(loser_color, self.cfg.REWARD_LOSE, terminal=True)
                    self._add_reward_to_memory(winner_color, self.cfg.REWARD_WIN, terminal=True)
                    
                    loser_key = 'red' if loser_color == self.cfg.COLOR_RED else 'black'
                    winner_key = 'red' if winner_color == self.cfg.COLOR_RED else 'black'
                    ep_rewards[loser_key] += self.cfg.REWARD_LOSE
                    ep_rewards[winner_key] += self.cfg.REWARD_WIN
                    
                    self.env.game_over = True
                    self.env.winner = winner_turn
                    break
                
                # ==========================================
                # [新增] 動態溫度控制：前 20 步探索，之後貪婪收尾
                # ==========================================
                current_temperature = 1.0 if step_count < 20 else 0.0

                print(f"Episode: {i_episode} | Step: {step_count} | Turn: {current_turn}")
                
                # ==========================================
                # [修改] 呼叫結合 MCTS 的 select_action
                # 傳入 self.env 供推演，並解包三個回傳值
                # ==========================================
                action, log_prob, mcts_probs = self.agent.select_action(
                    env=self.env, 
                    state=state, 
                    eaten_state=eaten_state, 
                    temperature=current_temperature
                )
                
                # 執行動作
                next_state_info, reward, done, info = self.env.step(action, i_episode)

                # 注意：請確保你的 env.step 與 env.reset 回傳的都是 (state_tensor, eaten_state_tensor) 兩個獨立物件
                next_state, next_eaten_state = next_state_info
                step_count += 1

                if done:
                    print(f"本局拿到的獎勵為: {reward}")

                # =============================================================
                # 儲存到正確的記憶體 (保持不變)
                # =============================================================
                current_color = self._get_player_color(current_turn)
                color_key = 'red' if current_color == self.cfg.COLOR_RED else 'black'
                ep_rewards[color_key] += reward  
                
                # [備註] 這裡我們仍然儲存 Neural Network 的 log_prob，
                # 這是因為 PPO 更新時需要算 Ratio，不能用 MCTS 的機率來算 Ratio。
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
                    # ... 依此類推，與你原本代碼相同 ...

                # 推進狀態
                state = next_state
                eaten_state = next_eaten_state

                # ... (後續的 info Eaten_reward 處理、遊戲結束勝負獎勵分配、PPO Update 邏輯都完全不用動) ...
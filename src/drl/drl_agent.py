"""Deep Reinforcement Learning (DRL) Prognostic Strategist (SIH26054).

Actor-Critic (PPO) agent for real-time aero piston engine prognostic intervention:
  - State Space (S): 15-dim state including PINN physical gradient + sensor streams + RUL.
  - Action Space (A): Continuous [delta_throttle, delta_mixture] recommendations.
  - PINN Safety Shield: Enforces zero-tolerance boundary violations during flight.
  - Generates actionable operational advice to prolong Remaining Useful Life.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .engine_env import AeroEngineEnv, CHT_CRIT, VIB_CRIT

ROOT = Path(__file__).resolve().parents[2]


class ActorCritic(nn.Module):
    """Continuous Actor-Critic Network."""

    def __init__(self, state_dim: int = 15, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy head (Actor)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim) - 1.0)

        # Value head (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor):
        feat = self.shared(state)
        mean = torch.tanh(self.actor_mean(feat)) * 0.15  # bound action magnitude
        std = torch.exp(self.actor_logstd).expand_as(mean)
        val = self.critic(feat)
        return Normal(mean, std), val

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        dist, val = self.forward(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, val


def train_drl_agent(episodes: int = 120, steps_per_rollout: int = 200, lr: float = 3e-4, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DRL Prognostic Strategist] Training on device: {device}")

    env = AeroEngineEnv(max_cycles=300)
    policy = ActorCritic(state_dim=15, action_dim=2, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    archetypes = ["cht_over", "vibration_over", "oil_starvation", "egt_over"]
    episode_rewards = []
    episode_durations = []
    shield_interventions = []

    for ep in range(episodes):
        arch = archetypes[ep % len(archetypes)]
        state = env.reset(failure_archetype=arch)
        ep_reward = 0.0

        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []

        for step in range(steps_per_rollout):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, val = policy.get_action(state_tensor)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done, info = env.step(action_np, use_shield=True)

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor([[reward]], dtype=torch.float32, device=device))
            values.append(val)
            masks.append(torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device))

            ep_reward += reward
            state = next_state
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_durations.append(env.current_cycle)
        shield_interventions.append(env.shield_interventions)

        # Policy update using GAE
        if len(states) > 5:
            with torch.no_grad():
                next_state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                _, next_val = policy(next_state_t)
                returns = []
                discounted_reward = next_val
                for r, m in zip(reversed(rewards), reversed(masks)):
                    discounted_reward = r + 0.98 * discounted_reward * m
                    returns.insert(0, discounted_reward)

            returns = torch.cat(returns)
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            states = torch.cat(states)
            actions = torch.cat(actions)

            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # PPO surrogate update
            dist, new_values = policy(states)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(new_log_probs - log_probs.detach())

            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns.detach())
            entropy_loss = -0.01 * dist.entropy().mean()

            total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        if (ep + 1) % 15 == 0 or ep == 0:
            avg_rew = np.mean(episode_rewards[-15:])
            avg_dur = np.mean(episode_durations[-15:])
            avg_shield = np.mean(shield_interventions[-15:])
            print(f"Episode {ep+1:3d}/{episodes} | Avg Reward: {avg_rew:6.1f} | Avg Flight Endurance: {avg_dur:5.1f} cycles | Shield Interventions: {avg_shield:.1f}")

    # Save checkpoint
    ckpt_dir = ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "state_dim": 15,
        "action_dim": 2,
        "avg_endurance": float(np.mean(episode_durations[-20:])),
    }, ckpt_dir / "drl_policy.pt")
    print(f"[DRL Prognostic Strategist] Checkpoint saved to {ckpt_dir / 'drl_policy.pt'}")

    # Plot training learning curves
    plot_drl_curves(episode_rewards, episode_durations, ROOT / "models" / "figures" / "drl_learning_curves.png")
    return policy


def plot_drl_curves(rewards, durations, save_path: Path):
    """Plot DRL flight extension learning curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

        ax1.plot(rewards, color="#0284c7", alpha=0.3, label="Raw Episode Reward")
        if len(rewards) >= 10:
            rolling_r = np.convolve(rewards, np.ones(10)/10, mode="valid")
            ax1.plot(range(9, len(rewards)), rolling_r, color="#0369a1", linewidth=2.0, label="10-Ep Rolling Mean")
        ax1.set_xlabel("Training Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Prognostic DRL Policy Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(durations, color="#10b981", alpha=0.3, label="Safe Flight Cycles")
        if len(durations) >= 10:
            rolling_d = np.convolve(durations, np.ones(10)/10, mode="valid")
            ax2.plot(range(9, len(durations)), rolling_d, color="#047857", linewidth=2.0, label="10-Ep Rolling Mean")
        ax2.set_xlabel("Training Episode")
        ax2.set_ylabel("Mission Endurance (Cycles)")
        ax2.set_title("MALE UAV Flight Endurance Extension via DRL")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[DRL] Learning curve saved to {save_path}")
    except Exception as e:
        print(f"[DRL] Plotting error: {e}")


def recommend_prognostic_action(policy: ActorCritic, state_vector: np.ndarray,
                                current_cht: float = 160.0, current_vib: float = 1.5,
                                current_rul: float = 200.0) -> dict:
    """Interface for Tool 2 (DRL RUL Core) in LangGraph & WebSocket Server."""
    policy.eval()
    with torch.no_grad():
        st = torch.from_numpy(state_vector).float().unsqueeze(0)
        dist, val = policy(st)
        action = dist.mean.numpy()[0]

    d_throttle = float(action[0])
    d_mixture = float(action[1])

    # PINN Safety boundary check
    shield_applied = False
    warning_flag = None

    if current_cht > 185.0 and d_throttle > 0:
        shield_applied = True
        d_throttle = min(d_throttle, -0.04)  # force throttle de-rate
        warning_flag = "PINN Shield: Blocked throttle surge due to elevated CHT (>185°C)"

    if current_vib > 2.8 and d_throttle > -0.02:
        shield_applied = True
        d_throttle = -0.06  # force vibration mitigation
        warning_flag = "PINN Shield: Enforced mechanical de-rate due to high vibration RMS (>2.8g)"

    # Projected RUL extension under policy
    if d_throttle < -0.02 or d_mixture < 0:
        projected_life_extension = max(15.0, (0.05 - d_throttle) * 180.0)
    else:
        projected_life_extension = 0.0

    recommendation = []
    if d_throttle < -0.02:
        recommendation.append(f"De-rate cruise throttle by {abs(d_throttle)*100:.1f}% to reduce thermal load")
    elif d_throttle > 0.02:
        recommendation.append(f"Trim throttle +{d_throttle*100:.1f}% for airspeed maintenance")
    else:
        recommendation.append("Maintain nominal cruise throttle")

    if d_mixture < -0.02:
        recommendation.append(f"Enrich fuel-air mixture (-{abs(d_mixture)*2:.1f} AFR) for cylinder cooling")
    elif d_mixture > 0.02:
        recommendation.append(f"Lean mixture (+{d_mixture*2:.1f} AFR) for fuel economy")

    return {
        "delta_throttle": d_throttle,
        "delta_mixture": d_mixture,
        "shield_applied": shield_applied,
        "warning_flag": warning_flag,
        "recommendation": " | ".join(recommendation),
        "projected_extension_cycles": round(projected_life_extension, 1),
        "adjusted_rul": round(current_rul + projected_life_extension, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    train_drl_agent(episodes=args.episodes)

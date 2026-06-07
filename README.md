[README_beer_game.md](https://github.com/user-attachments/files/28679751/README_beer_game.md)
# 🍺 Beer Distribution Game — Q-Learning RL Agent

A reinforcement learning simulation of the classic MIT Sloan Beer Distribution Game. A Q-learning agent is trained to manage retailer inventory decisions across a 4-stage supply chain, with the goal of minimizing the Bullwhip Effect.

## What is the Beer Game?

The Beer Distribution Game was developed at MIT Sloan in the 1960s to demonstrate how small demand fluctuations at the retail level cause increasingly wild order swings upstream — a phenomenon known as the **Bullwhip Effect**. It costs global supply chains an estimated **$50–100 billion annually** in excess inventory and lost sales.

FACTORY → DISTRIBUTOR → WHOLESALER → RETAILER → Customer
Each stage can only see its own inventory. No one sees the full picture. That information asymmetry is exactly what causes the chaos — and what the RL agent is trained to handle.

## Project Goal

Train a Q-learning agent to control the Retailer stage and make smarter ordering decisions than a human heuristic player — specifically, to place more stable orders that don't amplify upstream volatility.

## Reward Function

| Event | Reward |
|---|---|
| Meeting customer demand exactly | +20 |
| Partial fulfillment | proportional |
| Each unit of backlog | −2 per unit |
| Each unit held in inventory | −0.5 per unit |
| Stockout (zero stock, demand > 0) | −10 |
| Stable order (within ±2 of last week) | +2 anti-bullwhip bonus |

The anti-bullwhip bonus is the key design decision here — it directly incentivizes the agent to smooth its ordering behavior rather than panic-order when inventory drops.

---

## Results

Training ran for **120 episodes × 36 weeks** each. Results compared against a human-like heuristic baseline (20 episodes).

| Metric | Human Baseline | RL Agent (last 15 eps) |
|---|---|---|
| Avg Reward | +337.78 | −318.81 |
| Bullwhip Ratio σ(orders)/σ(demand) | 1.153 | 3.469 |
| Avg Total Backlog | 0.0 units | 143.2 units |
| Avg Stockout Weeks | N/A | 8.27 / 36 |
| Best Single Episode Reward | — | +284.32 |

**The agent did not outperform the human baseline within 120 episodes.** The bullwhip ratio of 3.469 indicates the agent was still amplifying demand variance rather than dampening it by the end of training. The best single episode (+284.32) shows the policy *can* work — but it hasn't converged consistently.

### Why it didn't converge

A few honest reasons:

- **120 episodes is too few.** The Q-table state space (inv × backlog × incoming × demand bins) has thousands of states. Most are visited only a handful of times during training.
- **Reward shaping tension.** The holding cost and backlog penalty pull in opposite directions. The agent oscillates between over-stocking and under-stocking rather than finding a stable middle ground.
- **Human baseline is deceptively strong.** The heuristic happens to match the demand pattern well (steady ~8 units/week after week 4), so the bar is higher than it looks.

## Simulation Parameters

```python
N_EPISODES   = 120     # training episodes
N_WEEKS      = 36      # weeks per episode (classic beer game length)
LEAD_TIME    = 2       # weeks for orders to arrive
ALPHA        = 0.15    # Q-learning rate
GAMMA        = 0.90    # discount factor
EPS_DECAY    = 0.97    # exploration decay per episode
MAX_ORDER    = 20      # max units agent can order per week
```

## Context

Built as part of a broader interest in applying RL to supply chain problems. The Beer Game is a well-studied environment and a natural starting point before moving to more complex multi-agent or continuous-action settings. The negative results here are documented intentionally — knowing *why* a baseline method fails is as useful as knowing that it does.

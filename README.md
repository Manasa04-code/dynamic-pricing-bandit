# Dynamic Pricing using Multi-Armed Bandits

This project implements a generic *dynamic pricing engine* using *Reinforcement Learning (Multi-Armed Bandit algorithms)* to learn the optimal selling price over time without labeled historical data.

---

## 🚀 Why Bandits for Pricing?

Traditional ML requires labeled data — in pricing we often *do not know in advance* which price will perform the best.  
A bandit model *learns while earning* — it simultaneously explores new prices and exploits the best observed one.

---

## 📌 Algorithms Implemented

- *ε-Greedy*
- *UCB (Upper Confidence Bound)*
- *Thompson Sampling*

All three are compared on the same environment to measure regret and revenue lift.

---

## 🧰 Tech Stack

- Python
- Numpy
- Jupyter Notebook (analysis & experiments)

(Flask / Streamlit deployment will be added later)

---

## 📂 Planned Folder Structure

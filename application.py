# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import time
import math
import random
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Dynamic Pricing — Bandit Playground")

# -----------------------
# Environment (online)
# -----------------------
class PricingEnv:
    """
    Online simulator: each step we sample a new customer context and return conversion
    probability for a chosen price (so chosen arm always receives feedback).
    """
    def __init__(self, price_candidates, seed=None):
        self.price_candidates = list(price_candidates)
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        # product/customer generative parameters
        self.base_value_mean = 50.0
        self.base_value_std = 8.0

    def sample_customer(self):
        # sample a base value and a segment multiplier
        base_value = float(self.rng.normal(self.base_value_mean, self.base_value_std))
        seg_alpha = float(self.rng.choice([0.8, 1.0, 1.2], p=[0.4,0.45,0.15]))
        day = int(self.rng.randint(0,7))
        day_sin = math.sin(2*math.pi*day/7)
        recent_trend = float(self.rng.normal(0, 0.05))
        context = np.array([base_value, seg_alpha, day_sin, recent_trend])
        return context

    def purchase_probability(self, context, price, competitor_price=0.0):
        base_value, alpha, day_sin, recent_trend = context
        value = base_value * alpha
        comp_diff = competitor_price - price
        k = 0.12
        gamma = 0.02
        score = k*(value - price) + gamma*comp_diff + 0.5*day_sin + recent_trend
        prob = 1.0 / (1.0 + math.exp(-score))
        return float(np.clip(prob, 0.0, 1.0))

    def step(self, chosen_price):
        ctx = self.sample_customer()
        comp_price = float(self.rng.normal(np.median(self.price_candidates), 2.0))
        p_buy = self.purchase_probability(ctx, chosen_price, comp_price)
        bought = self.rng.rand() < p_buy
        revenue = chosen_price * (1 if bought else 0)
        return {
            "context": ctx,
            "competitor_price": comp_price,
            "p_buy": p_buy,
            "bought": int(bought),
            "revenue": float(revenue)
        }

# -----------------------
# Bandit implementations (online-friendly)
# -----------------------
class EpsilonGreedy:
    def __init__(self, prices, eps=0.1, seed=None):
        self.prices = list(prices)
        self.eps = eps
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)
        self.rng = np.random.RandomState(seed if seed else 123)

    def choose(self):
        if self.rng.rand() < self.eps:
            return float(self.rng.choice(self.prices))
        avg = {p: (self.rewards[p] / self.counts[p]) if self.counts[p]>0 else 0.0 for p in self.prices}
        return max(self.prices, key=lambda p: avg[p])

    def update(self, price, reward):
        self.counts[price]+=1
        self.rewards[price]+=reward

    def avg_rewards(self):
        return {p: (self.rewards[p]/self.counts[p]) if self.counts[p]>0 else 0.0 for p in self.prices}

class UCB1:
    def __init__(self, prices, seed=None):
        self.prices = list(prices)
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)
        self.t = 0
        self.rng = np.random.RandomState(seed if seed else 123)

    def choose(self):
        self.t += 1
        # try untried arms first
        for p in self.prices:
            if self.counts[p] == 0:
                return p
        ucb_vals = {}
        for p in self.prices:
            mean = self.rewards[p] / self.counts[p]
            bonus = math.sqrt(2 * math.log(max(1, self.t)) / self.counts[p])
            ucb_vals[p] = mean + bonus
        return max(self.prices, key=lambda p: ucb_vals[p])

    def update(self, price, reward):
        self.counts[price]+=1
        self.rewards[price]+=reward

    def avg_rewards(self):
        return {p: (self.rewards[p]/self.counts[p]) if self.counts[p]>0 else 0.0 for p in self.prices}

class ThompsonGaussian:
    def __init__(self, prices, seed=None, min_sigma=1e-2):
        self.prices = list(prices)
        self.counts = defaultdict(int)
        self.sum_rewards = defaultdict(float)
        self.sum_sqs = defaultdict(float)
        self.rng = np.random.RandomState(seed if seed else 123)
        self.min_sigma = min_sigma

    def choose(self):
        samples = {}
        for p in self.prices:
            n = self.counts[p]
            if n == 0:
                samples[p] = self.rng.rand() + max(self.prices)  # encourage exploration early
            else:
                mean = self.sum_rewards[p] / n
                var = max((self.sum_sqs[p]/n) - mean**2, 1e-6)
                stderr = math.sqrt(var) / math.sqrt(n + 1)
                samples[p] = self.rng.normal(mean, stderr if stderr>0 else self.min_sigma)
        return max(self.prices, key=lambda p: samples[p])

    def update(self, price, reward):
        self.counts[price]+=1
        self.sum_rewards[price]+=reward
        self.sum_sqs[price]+=reward*reward

    def avg_rewards(self):
        return {p: (self.sum_rewards[p]/self.counts[p]) if self.counts[p]>0 else 0.0 for p in self.prices}

# Random baseline
class RandomPolicy:
    def __init__(self, prices, seed=None):
        self.prices = list(prices)
        self.rng = np.random.RandomState(seed if seed else 0)
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)

    def choose(self):
        return float(self.rng.choice(self.prices))

    def update(self, price, reward):
        self.counts[price]+=1
        self.rewards[price]+=reward

    def avg_rewards(self):
        return {p: (self.rewards[p]/self.counts[p]) if self.counts[p]>0 else 0.0 for p in self.prices}

# -----------------------
# Streamlit UI
# -----------------------
st.title("Dynamic Pricing — Bandits Playground (Online Simulator)")

# sidebar controls (minimal)
with st.sidebar:
    st.header("Controls")
    price_candidates = st.multiselect("Price candidates", options=[30,40,50,60,70,80,90,100,120,140,160], default=[100,120,140,160])
    if len(price_candidates) < 2:
        st.warning("Pick at least 2 price candidates")
    policy_name = st.selectbox("Policy", options=["Epsilon-Greedy","UCB1","Thompson (Gaussian)","Random"])
    eps = st.slider("ε (for ε-greedy)", 0.0, 0.5, 0.1, 0.01)
    n_steps = st.number_input("Total steps (max when Play runs)", min_value=10, max_value=5000, value=200, step=10)
    step_delay = st.slider("Play delay (seconds)", 0.0, 1.5, 0.15, 0.05)

# main layout: left controls, right charts
left, right = st.columns([1,2])

# session state init
if "sim_state" not in st.session_state:
    st.session_state.sim_state = {
        "env": None,
        "policy": None,
        "history": [],
        "running": False,
        "step_idx": 0
    }

def init_sim():
    env = PricingEnv(price_candidates, seed=42)
    if policy_name == "Epsilon-Greedy":
        policy = EpsilonGreedy(price_candidates, eps=eps, seed=1)
    elif policy_name == "UCB1":
        policy = UCB1(price_candidates, seed=1)
    elif policy_name == "Thompson (Gaussian)":
        policy = ThompsonGaussian(price_candidates, seed=1)
    else:
        policy = RandomPolicy(price_candidates, seed=1)
    st.session_state.sim_state.update({
        "env": env,
        "policy": policy,
        "history": [],
        "running": False,
        "step_idx": 0
    })

def reset_sim():
    init_sim()

with left:
    st.subheader("Run controls")
    if st.button("Init / Reset"):
        reset_sim()
        st.success("Simulator initialized")
    step_col1, step_col2 = st.columns(2)
    if step_col1.button("Step"):
        # run a single step
        if st.session_state.sim_state["policy"] is None:
            init_sim()
        env = st.session_state.sim_state["env"]
        policy = st.session_state.sim_state["policy"]
        chosen = policy.choose()
        res = env.step(chosen)
        policy.update(chosen, res["revenue"])
        st.session_state.sim_state["history"].append({
            "step": st.session_state.sim_state["step_idx"],
            "price": chosen,
            "revenue": res["revenue"],
            "bought": res["bought"],
            "p_buy": res["p_buy"]
        })
        st.session_state.sim_state["step_idx"] += 1

    play = st.button("Play / Run Full")
    stop = st.button("Stop (if playing)")

    # handle play behaviour
    if play:
        if st.session_state.sim_state["policy"] is None:
            init_sim()
        st.session_state.sim_state["running"] = True

    if stop:
        st.session_state.sim_state["running"] = False

    # optionally run while loop (Play)
    if st.session_state.sim_state["running"]:
        # run until step_idx reaches n_steps or user stops
        while st.session_state.sim_state["running"] and st.session_state.sim_state["step_idx"] < n_steps:
            env = st.session_state.sim_state["env"]
            policy = st.session_state.sim_state["policy"]
            chosen = policy.choose()
            res = env.step(chosen)
            policy.update(chosen, res["revenue"])
            st.session_state.sim_state["history"].append({
                "step": st.session_state.sim_state["step_idx"],
                "price": chosen,
                "revenue": res["revenue"],
                "bought": res["bought"],
                "p_buy": res["p_buy"]
            })
            st.session_state.sim_state["step_idx"] += 1
            # update visuals incrementally by short pause
            time.sleep(step_delay)
            # allow user to break by pressing Stop (button click is handled on next rerun)
            # streamlit will re-run the script; if Stop clicked, running flag will be reset
            # break control: if running remains True, continue

with right:
    st.subheader("Metrics & Charts")
    history = st.session_state.sim_state["history"]
    if not history:
        st.info("No steps yet — press 'Init / Reset' then 'Step' or 'Play / Run Full'.")
    else:
        df_hist = pd.DataFrame(history)
        # KPI metrics
        total_rev = df_hist['revenue'].sum()
        total_conv = df_hist['bought'].sum()
        st.metric("Cumulative revenue", f"{total_rev:.2f}")
        st.metric("Cumulative conversions", f"{int(total_conv)}")

        # cumulative revenue plot
        df_hist['cum_revenue'] = df_hist['revenue'].cumsum()
        st.line_chart(df_hist[['cum_revenue']].rename(columns={'cum_revenue':'Cumulative Revenue'}))

        # price distribution
        st.subheader("Price distribution (counts)")
        price_counts = df_hist['price'].value_counts().sort_index()
        st.bar_chart(price_counts)

        # average reward per price (observed)
        st.subheader("Average observed reward per price")
        avg_reward = {}
        for p in price_candidates:
            sel = df_hist[df_hist['price'] == p]
            avg_reward[p] = sel['revenue'].mean() if len(sel) > 0 else 0.0
        avg_df = pd.DataFrame({
            "price": list(avg_reward.keys()),
            "avg_reward": list(avg_reward.values())
        }).set_index('price')
        st.bar_chart(avg_df)

        # show last N rows
        st.subheader("Latest events")
        st.dataframe(df_hist.tail(30).reset_index(drop=True))

# Footer / quick tips
st.markdown("---")
st.markdown("**Tips:** Use small step_delay (0.05–0.2) for smoother Play. Use `Init / Reset` after changing price set or policy.")

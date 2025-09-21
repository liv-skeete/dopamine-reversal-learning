#!/usr/bin/env python3
"""
Incentive Salience Experiment Runner
====================================

Main script to run incentive salience experiments from the original Colab code.
"""

import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.incentive_salience_agent import IncentiveSalienceAgent
from src.utils.helpers import get_object_index, get_outcome_index

# Experiment constants
NUM_OBJECTS = 3
SESSIONS = 10
TRIALS_ACQ = 10
TRIALS_REV = 20
TRIALS_PER_SESSION = TRIALS_ACQ + TRIALS_REV
GROUP_ORDER = ["Normal", "Addicted"]

def run_session(agent, track_meta=False):
    """Run a single session of the incentive salience experiment."""
    out = {"acquisition_errors": 0, "reversal_errors": 0, "perseverative_errors": 0}
    meta_choice_log, q_acc_log, v_acc_log = [], [], []
    v_offpolicy_acc_log, q_offpolicy_acc_log = [], []
    agent.trial = 0
    agent.reset_session()
    correct_obj = random.choice([np.eye(3)[i] for i in range(3)])
    prev_obj = None
    
    for trial in range(TRIALS_PER_SESSION):
        if trial == TRIALS_ACQ:
            agent.reset_session()
            prev_obj = correct_obj
            correct_obj = random.choice([np.eye(3)[i] for i in range(3) 
                                       if not np.array_equal(np.eye(3)[i], prev_obj)])
        
        objs = [np.eye(3)[i] for i in random.sample(range(3), 3)]
        a, used_v = agent.choose(objs, correct_obj)
        choice = objs[a]
        reward = int(np.array_equal(choice, correct_obj))
        agent.update(a, objs, reward, correct_obj)
        agent.trial += 1
        meta_choice_log.append(used_v)
        q_acc_log.append(reward if not used_v else np.nan)
        v_acc_log.append(reward if used_v else np.nan)

        # Off-policy accuracy calculations with salience
        v_vals = []
        for i, obj in enumerate(objs):
            obj_id = get_object_index(obj)
            loc = i
            rew = int(np.array_equal(obj, correct_obj))
            v_idx = get_outcome_index(loc, obj_id, rew)
            v_val = agent.V[v_idx]
            if obj_id == 0:  # Apply salience bonus to obj0
                v_val += agent.salience_bonus
            v_vals.append(v_val)
        v_pred_choice = np.argmax(v_vals)
        v_offpolicy_acc_log.append(1 if np.array_equal(objs[v_pred_choice], correct_obj) else 0)
        
        q_vals = agent.Q[[get_object_index(obj) for obj in objs]].copy()
        for i, obj_id in enumerate([get_object_index(obj) for obj in objs]):
            if obj_id == 0:  # Apply salience bonus to obj0
                q_vals[i] += agent.salience_bonus
        q_pred_choice = np.argmax(q_vals)
        q_offpolicy_acc_log.append(1 if np.array_equal(objs[q_pred_choice], correct_obj) else 0)

        # Meta-policy switching
        if agent.meta_mode == "V" and used_v:
            agent.update_meta_policy(reward)

        # Error counting
        if trial < TRIALS_ACQ:
            if not reward:
                out["acquisition_errors"] += 1
        else:
            if np.array_equal(choice, prev_obj):
                out["perseverative_errors"] += 1
            if not reward:
                out["reversal_errors"] += 1

    if track_meta:
        return out, {'meta': meta_choice_log,
                     'q_acc': q_acc_log,
                     'v_acc': v_acc_log,
                     'v_offpolicy_acc': v_offpolicy_acc_log,
                     'q_offpolicy_acc': q_offpolicy_acc_log}
    else:
        return out

def run_group(name, bias, n=10, track_meta=False):
    """Run experiment for a group of agents."""
    group_data = []
    meta_logs = None
    for agent_idx in range(n):
        agent = IncentiveSalienceAgent(name, bias)
        record = {"acquisition_errors": [], "reversal_errors": [], "perseverative_errors": []}
        meta_trial_log = {"meta": [], "q_acc": [], "v_acc": [],
                          "v_offpolicy_acc": [], "q_offpolicy_acc": []}
        for session_idx in range(SESSIONS):
            agent.reset_session()
            if track_meta and agent_idx == 0 and session_idx == 0:
                res, trial_logs = run_session(agent, track_meta=True)
                for k in meta_trial_log:
                    meta_trial_log[k].extend(trial_logs[k])
            else:
                res = run_session(agent)
            for k in record:
                record[k].append(res[k])
        row = {k: np.mean(v) for k, v in record.items()}
        row["Group"] = name
        group_data.append(row)
        if track_meta and agent_idx == 0:
            meta_logs = meta_trial_log
    return pd.DataFrame(group_data), meta_logs

def plot_panel(df, metric, ax, ylabel):
    """Plot a panel for error metrics."""
    sns.barplot(data=df, x="Group", y=metric, order=GROUP_ORDER,
                estimator=np.mean, errorbar=("ci", 68), palette="muted",
                err_kws={'color':'black'}, ax=ax)
    means = df.groupby("Group")[metric].mean().reindex(GROUP_ORDER)
    for i, val in enumerate(means):
        ax.text(i, val + 0.4, f"{val:.1f}", ha='center', fontweight='bold')
    g1, g2 = (df[df.Group == g][metric] for g in GROUP_ORDER)
    y = means.max() + 1.8
    
    def sig(p, x1, x2):
        if p < 0.05:
            label = "**" if p < 0.01 else "*"
            ax.plot([x1, x1, x2, x2], [y, y + 0.2, y + 0.2, y], lw=1.3, color='black')
            ax.text((x1 + x2)/2, y + 0.3, label, ha='center', fontsize=12)
    
    sig(ttest_ind(g1, g2).pvalue, 0, 1)
    for i, grp in enumerate(GROUP_ORDER):
        ax.text(i, -1.5, f"N = {df[df.Group==grp].shape[0]}", ha='center', fontsize=9)
    ax.set_title(ylabel)
    ax.set_ylim(0, y + 2)
    ax.set_ylabel("")
    ax.set_xlabel("")

def main():
    """Run the main incentive salience experiment."""
    print("Running Incentive Salience Experiment...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run experiments
    normal_df, meta_normal = run_group("Normal", 0.8, 10, track_meta=True)
    addicted_df, _ = run_group("Addicted", 0.8, 12, track_meta=False)
    df = pd.concat([normal_df, addicted_df])
    
    # Create meta-strategy plot
    meta_matrix = np.zeros((10, TRIALS_PER_SESSION))
    v_off_acc_matrix = np.zeros((10, TRIALS_PER_SESSION))
    q_off_acc_matrix = np.zeros((10, TRIALS_PER_SESSION))
    
    for i in range(10):
        _, logs = run_group("Normal", 0.8, 1, track_meta=True)
        meta_matrix[i, :] = logs['meta'][:TRIALS_PER_SESSION]
        v_off_acc_matrix[i, :] = logs['v_offpolicy_acc'][:TRIALS_PER_SESSION]
        q_off_acc_matrix[i, :] = logs['q_offpolicy_acc'][:TRIALS_PER_SESSION]
    
    # Smoothing
    window = 3
    meta_mean = pd.Series(np.nanmean(meta_matrix, axis=0)).rolling(window, min_periods=1).mean()
    v_off_mean = pd.Series(np.nanmean(v_off_acc_matrix, axis=0)).rolling(window, min_periods=1).mean()
    q_off_mean = pd.Series(np.nanmean(q_off_acc_matrix, axis=0)).rolling(window, min_periods=1).mean()
    
    # Plot meta-strategy
    plt.figure(figsize=(10, 4))
    plt.plot(meta_mean, label="Avg V Used % (Normal)", color="green", lw=2)
    plt.plot(q_off_mean, '--', label="Off-policy Q acc (Normal)", color="purple", alpha=0.8)
    plt.plot(v_off_mean, ':', label="Off-policy V acc (Normal)", color="orange", alpha=0.9)
    plt.axvline(x=TRIALS_ACQ, color='red', linestyle=':', lw=2, label="Reversal")
    plt.ylabel("Fraction of Trials / Off-policy Accuracy")
    plt.xlabel("Trial (Session: 0-9 acquisition, 10-29 reversal)")
    plt.ylim(-0.02, 1.05)
    plt.title("Meta-Strategy Use & Off-policy Q/V Accuracy (Normal Only, N=10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/incentive_salience_meta_strategy.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot error panels
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    plot_panel(df, "acquisition_errors", axs[0], "Acquisition Errors (10)")
    plot_panel(df, "reversal_errors", axs[1], "Reversal Errors (20)")
    plot_panel(df, "perseverative_errors", axs[2], "Perseverative Errors (20)")
    plt.suptitle("Incentive Salience Theory — Hybrid (V→Q) vs. Addicted (Q-only)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../results/incentive_salience_errors.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    df.to_csv("../results/incentive_salience_results.csv", index=False)
    print("Results saved to ../results/")
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
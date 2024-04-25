import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score

tau = 0.01

if __name__ == "__main__":
    history = json.load(open("./experiment_history.json"))

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    s1_swing_cycles, s1_stance_cycles = history["s1_swing_cycles"], history["s1_stance_cycles"]
    s2_swing_cycles, s2_stance_cycles = history["s2_swing_cycles"], history["s2_stance_cycles"]
    s3_swing_cycles, s3_stance_cycles = history["s3_swing_cycles"], history["s3_stance_cycles"]
    s4_swing_cycles, s4_stance_cycles = history["s4_swing_cycles"], history["s4_stance_cycles"]

    swing_cycles_duration1 = [(right - left) / 1000
                              for left, right in s1_swing_cycles]

    stance_cycles_duration1 = [(right - left) / 1000
                              for left, right in s1_stance_cycles]

    swing_cycles_duration2 = [(right - left) / 1000
                              for left, right in s2_swing_cycles]

    stance_cycles_duration2 = [(right - left) / 1000
                               for left, right in s2_stance_cycles]

    swing_cycles_duration3 = [(right - left) / 1000
                              for left, right in s3_swing_cycles]

    stance_cycles_duration3 = [(right - left) / 1000
                               for left, right in s3_stance_cycles]

    swing_cycles_duration4 = [(right - left) / 1000
                              for left, right in s4_swing_cycles]

    stance_cycles_duration4 = [(right - left) / 1000
                               for left, right in s4_stance_cycles]

    combined_cycles1 = np.array(swing_cycles_duration1) + np.array(stance_cycles_duration1)
    combined_cycles2 = np.array(swing_cycles_duration2) + np.array(stance_cycles_duration2)
    combined_cycles3 = np.array(swing_cycles_duration3) + np.array(stance_cycles_duration3)
    combined_cycles4 = np.array(swing_cycles_duration4) + np.array(stance_cycles_duration4)

    true_swing_duration1 = history["true_s1_swing_duration"]
    true_stance_duration1 = history["true_s1_stance_duration"]

    true_swing_duration2 = history["true_s2_swing_duration"]
    true_stance_duration2 = history["true_s2_stance_duration"]

    true_swing_duration3 = history["true_s3_swing_duration"]
    true_stance_duration3 = history["true_s3_stance_duration"]

    true_swing_duration4 = history["true_s4_swing_duration"]
    true_stance_duration4 = history["true_s4_stance_duration"]

    swing_r2_state1 = r2_score(true_swing_duration1, swing_cycles_duration1)
    stance_r2_state1 = r2_score(true_stance_duration1, stance_cycles_duration1)

    swing_r2_state2 = r2_score(true_swing_duration2, swing_cycles_duration2)
    stance_r2_state2 = r2_score(true_stance_duration2, stance_cycles_duration2)

    swing_r2_state3 = r2_score(true_swing_duration3, swing_cycles_duration3)
    stance_r2_state3 = r2_score(true_stance_duration3, stance_cycles_duration3)

    swing_r2_state4 = r2_score(true_swing_duration4, swing_cycles_duration4)
    stance_r2_state4 = r2_score(true_stance_duration4, stance_cycles_duration4)

    fig.suptitle(f"A", fontsize=24)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    r2_swing_name = "$R^{2}_{swing}$"
    r2_stance_name = "$R^{2}_{stance}$"

    ax.text(0.05, 0.95, f"{r2_swing_name} = {swing_r2_state1:.3f}\n{r2_stance_name} = {stance_r2_state1:.3f}",
            transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    ax.plot(combined_cycles1, swing_cycles_duration1, color='#0081D9', linestyle='dashed',
     marker='o', linewidth=1, label='Swing')
    ax.plot(combined_cycles1, true_swing_duration1,
        color='black', linewidth=2, label="Halbertsma best-fit")

    ax.plot(combined_cycles1, stance_cycles_duration1, color='#F8550D',
        linestyle='dashed', marker='o', linewidth=1, label='Stance')
    ax.plot(combined_cycles1, true_stance_duration1,
        color='black', linewidth=2)

    ax.set_xlabel('Cycle duration, s', labelpad=10, fontsize=20)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2])
    ax.set_xlim([0, 2])

    ax.set_ylabel('Phase duration, s', labelpad=10, fontsize=20)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2])
    ax.set_ylim([0, 2])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

    fig.tight_layout()

    os.makedirs("./images", exist_ok=True)

    f_name = f"./images/phase_durations_error"
    plt.savefig(f_name + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(f_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)

    plt.show()

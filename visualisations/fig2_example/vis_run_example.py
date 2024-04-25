import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    name = "experiment_history"

    history = json.load(open(f"{name}.json"))

    history_len = int(len(history["swing1_state"]) / 1000)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    for i, j in [[16, 20], [36, 40], [76, 80]]:
        start = i * 1000
        end = j * 1000
        times = np.array(list(range(start, end))) / 1000

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

        fig.suptitle(f"Model simulation from {i} to {j} seconds", fontsize=24)

        axes[0].plot(times, history["swing1_state"][start:end], linewidth=2, color="#0081D9", label='Swing')
        axes[0].plot(times, history["stance1_state"][start:end], linewidth=2, color="#F8550D", label='Stance')
        axes[0].set_yticks([0, 1])
        axes[0].set_ylabel('Activation', labelpad=10, fontsize=20)
        axes[0].set_xticks([])
        axes[0].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

        axes[1].plot(times, history["swing2_state"][start:end], linewidth=2, color="#0081D9", label='Swing')
        axes[1].plot(times, history["stance2_state"][start:end], linewidth=2, color="#F8550D", label='Stance')
        axes[1].set_yticks([0, 1])
        axes[1].set_xlabel('Time, s', labelpad=10)
        axes[1].set_ylabel('Activation', labelpad=10, fontsize=20)

        axes[1].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

        axes[2].plot(times, history["swing3_state"][start:end], linewidth=2, color="#0081D9", label='Swing')
        axes[2].plot(times, history["stance3_state"][start:end], linewidth=2, color="#F8550D", label='Stance')
        axes[2].set_yticks([0, 1])
        axes[2].set_xlabel('Time, s', labelpad=10)
        axes[2].set_ylabel('Activation', labelpad=10, fontsize=20)

        axes[2].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

        axes[3].plot(times, history["swing4_state"][start:end], linewidth=2, color="#0081D9", label='Swing')
        axes[3].plot(times, history["stance4_state"][start:end], linewidth=2, color="#F8550D", label='Stance')
        axes[3].set_yticks([0, 1])
        axes[3].set_xlabel('Time, s', labelpad=10)
        axes[3].set_ylabel('Activation', labelpad=10, fontsize=20)

        axes[3].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

        fig.tight_layout()

        os.makedirs(f"./experiment_images", exist_ok=True)
        f_name = f"./experiment_images/{name} {i} to {j} seconds"
        # plt.savefig(f_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)
        plt.savefig(f_name + ".png", dpi=200, bbox_inches="tight")

        # plt.show()
        plt.close()

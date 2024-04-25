import json

import sys

sys.path.insert(0, "../../")
import optimize
import numpy as np
import tune_optimize_utils as utils

if __name__ == "__main__":
    history, error, error_phase, _, _, _ = optimize.simulation_error(params=utils.best_params[0],
                                                                     progress_bar=True
                                                                     )

    print("error ", error)
    print("error_phase ", error_phase)

    s1_swing_cycles, s1_stance_cycles = optimize.calc_swing_stance(history["s1_state"])
    s2_swing_cycles, s2_stance_cycles = optimize.calc_swing_stance(history["s2_state"])
    s3_swing_cycles, s3_stance_cycles = optimize.calc_swing_stance(history["s3_state"])
    s4_swing_cycles, s4_stance_cycles = optimize.calc_swing_stance(history["s4_state"])

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
    true_s1_swing_duration = optimize.cycle_to_swing(combined_cycles1)
    true_s1_stance_duration = optimize.cycle_to_stance(combined_cycles1)

    combined_cycles2 = np.array(swing_cycles_duration2) + np.array(stance_cycles_duration2)
    true_s2_swing_duration = optimize.cycle_to_swing(combined_cycles2)
    true_s2_stance_duration = optimize.cycle_to_stance(combined_cycles2)

    combined_cycles3 = np.array(swing_cycles_duration3) + np.array(stance_cycles_duration3)
    true_s3_swing_duration = optimize.cycle_to_swing(combined_cycles3)
    true_s3_stance_duration = optimize.cycle_to_stance(combined_cycles3)

    combined_cycles4 = np.array(swing_cycles_duration4) + np.array(stance_cycles_duration4)
    true_s4_swing_duration = optimize.cycle_to_swing(combined_cycles4)
    true_s4_stance_duration = optimize.cycle_to_stance(combined_cycles4)

    for k in history.keys():
        history[k] = history[k][:, 0].tolist()

    history["s1_swing_cycles"] = s1_swing_cycles
    history["s1_stance_cycles"] = s1_stance_cycles
    history["s2_swing_cycles"] = s2_swing_cycles
    history["s2_stance_cycles"] = s2_stance_cycles
    history["s3_swing_cycles"] = s3_swing_cycles
    history["s3_stance_cycles"] = s3_stance_cycles
    history["s4_swing_cycles"] = s4_swing_cycles
    history["s4_stance_cycles"] = s4_stance_cycles
    history["true_s1_swing_duration"] = true_s1_swing_duration.tolist()
    history["true_s1_stance_duration"] = true_s1_stance_duration.tolist()
    history["true_s2_swing_duration"] = true_s2_swing_duration.tolist()
    history["true_s2_stance_duration"] = true_s2_stance_duration.tolist()
    history["true_s3_swing_duration"] = true_s3_swing_duration.tolist()
    history["true_s3_stance_duration"] = true_s3_stance_duration.tolist()
    history["true_s4_swing_duration"] = true_s4_swing_duration.tolist()
    history["true_s4_stance_duration"] = true_s4_stance_duration.tolist()

    json.dump(history, open("experiment_history.json", 'w'))

import numpy as np
from sklearn.metrics import mean_squared_error
import nengo

try:
    import nengo_ocl
except:
    print("Warning: can't import nengo_ocl. Using CPU")
from cpg import create_CPG

tau = 0.01

# Halbertsma minmax(Tc)
# Minimum and maximum ranges for cat cycle duration
MIN_PHASE, MAX_PHASE = (.57, 1.91)


def findall(p, s):
    """Yields all the positions of
    the pattern p in the string s."""
    i = s.find(p)
    r = []
    while i != -1:
        r.append(i)
        i = s.find(p, i + 1)
    return r


def calc_swing_stance(state_probe):
    """
    Function detects starts and ends for swing and stance phases
    using findall function
    giving state changes history
    """
    s1_state_changes = state_probe < 0

    state_str = "".join([str(int(s)) for s in s1_state_changes])

    stance_swing = findall("01", state_str)
    swing_stance = findall("10", state_str)

    swing_cycles = []
    stance_cycles = []

    for start_swing in stance_swing:
        start_stance_i = np.searchsorted(swing_stance, start_swing)
        if start_stance_i >= len(swing_stance):
            break
        start_stance = swing_stance[start_stance_i]

        end_stance_i = np.searchsorted(stance_swing, start_stance)
        if end_stance_i >= len(stance_swing):
            end_stance = len(state_probe)
        else:
            end_stance = stance_swing[end_stance_i]

        swing_cycles.append((start_swing, start_stance))
        stance_cycles.append((start_stance, end_stance))

    return swing_cycles[1:-1], stance_cycles[1:-1]


def cycle_to_swing(cycle):
    """
    Calculates expected swing phase duration given cycle duration
    as present in Halbertsma cats dataset
    """
    return 0.168 + 0.0938 * cycle


def cycle_to_stance(cycle):
    """
    Calculates expected stance phase duration given cycle duration
    as present in Halbertsma cats dataset
    """
    return -0.168 + 0.9062 * cycle


def single_limb_error(swing_cycles, stance_cycles):
    """
    Function computers phase duration loss for one limb
    by calculating root-mean-square error of expected phase durations and simulated phases
    Our goal to reproduce the same linear relationship present in Halbertsma dataset
    """
    swing_cycles_duration = [(right - left) / 1000
                             for left, right in swing_cycles]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in stance_cycles]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

    swing_expected = cycle_to_swing(combined_cycles)
    stance_expected = cycle_to_stance(combined_cycles)
    err_swing = mean_squared_error(swing_expected,
                                   swing_cycles_duration, squared=False
                                   )
    err_stance = mean_squared_error(stance_expected,
                                    stance_cycles_duration, squared=False
                                    )
    error_phase = err_swing + err_stance

    error_speed = abs(MIN_PHASE - min(combined_cycles)) + \
                  abs(MAX_PHASE - max(combined_cycles))

    return error_phase, error_speed


def symmetry_error(swing_cycles, stance_cycles):
    """
    Function checks if swing phase of a limb is in the middle of stance phase in other limb
    """
    pre_swing_part = [abs(swing[0] - stance[0]) / 1000
                      for swing, stance in zip(swing_cycles, stance_cycles)]

    post_swing_part = [abs(stance[1] - swing[1]) / 1000
                       for swing, stance in zip(swing_cycles, stance_cycles)]

    error = mean_squared_error(pre_swing_part, post_swing_part, squared=False)

    return error


def simulation(params, time=80, progress_bar=False, state_neurons=300, **args):
    """
    Function creates CPG model given parameters and run simulation
    Returns dictionary for simulation history
    """
    model = create_CPG(params=params, state_neurons=state_neurons, time=time, **args)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)
        s3_probe = nengo.Probe(model.s3, synapse=tau)
        s4_probe = nengo.Probe(model.s4, synapse=tau)
        speed_probe = nengo.Probe(model.speed, synapse=tau)

        swing1_probe = nengo.Probe(model.swing1, synapse=tau)
        stance1_probe = nengo.Probe(model.stance1, synapse=tau)
        swing2_probe = nengo.Probe(model.swing2, synapse=tau)
        stance2_probe = nengo.Probe(model.stance2, synapse=tau)

        ############################################################
        swing3_probe = nengo.Probe(model.swing3, synapse=tau)
        stance3_probe = nengo.Probe(model.stance3, synapse=tau)
        swing4_probe = nengo.Probe(model.swing4, synapse=tau)
        stance4_probe = nengo.Probe(model.stance4, synapse=tau)
        ############################################################

    with nengo.Simulator(model, progress_bar=progress_bar, optimize=True) as sim:
        sim.run(time)

    return {
        "s1_state": sim.data[s1_probe],
        "s2_state": sim.data[s2_probe],
        "s3_state": sim.data[s3_probe],
        "s4_state": sim.data[s4_probe],
        "speed_state": sim.data[speed_probe],
        "swing1_state": sim.data[swing1_probe],
        "stance1_state": sim.data[stance1_probe],
        "swing2_state": sim.data[swing2_probe],
        "stance2_state": sim.data[stance2_probe],
        "swing3_state": sim.data[swing3_probe],
        "stance3_state": sim.data[stance3_probe],
        "swing4_state": sim.data[swing4_probe],
        "stance4_state": sim.data[stance4_probe],
    }


def simulation_error(params, time=80, progress_bar=False, state_neurons=300, **args):
    """
    Function runs simulation and combines all losses into final value
    """
    history = simulation(params, time, progress_bar, state_neurons=state_neurons, **args)

    s1_state = history["s1_state"]
    s2_state = history["s2_state"]
    ##############################
    s3_state = history["s3_state"]
    s4_state = history["s4_state"]
    ##############################

    try:
        sw_cycles1_l, st_cycles1_l = calc_swing_stance(s1_state)
        sw_cycles2_r, st_cycles2_r = calc_swing_stance(s2_state)

        ########################################################
        sw_cycles4_l, st_cycles4_l = calc_swing_stance(s3_state)
        sw_cycles3_r, st_cycles3_r = calc_swing_stance(s4_state)
        ########################################################

        error_phase1_l, error_speed1_l = single_limb_error(sw_cycles1_l, st_cycles1_l)
        error_phase2_r, error_speed2_r = single_limb_error(sw_cycles2_r, st_cycles2_r)

        #######################################################################################################
        error_phase4_l, error_speed4_l = single_limb_error(sw_cycles4_l, st_cycles4_l)
        error_phase3_r, error_speed3_r = single_limb_error(sw_cycles3_r, st_cycles3_r)
        #######################################################################################################

        error_phase1 = error_phase1_l + error_phase2_r
        error_phase2 = error_phase4_l + error_speed3_r

        error_speed1 = error_speed1_l + error_speed2_r
        error_speed2 = error_speed4_l + error_speed3_r

        error_phase = error_phase1 + error_phase2
        error_speed = error_speed1 + error_speed2

        error_symmetricity_1_2 = symmetry_error(sw_cycles1_l[1:], st_cycles2_r)
        error_symmetricity_2_1 = symmetry_error(sw_cycles2_r, st_cycles1_l[:-1])

        #############################################################################################################
        error_symmetricity_4_3 = symmetry_error(sw_cycles4_l[1:], st_cycles3_r)
        error_symmetricity_3_4 = symmetry_error(sw_cycles3_r, st_cycles4_l[:-1])

        error_symmetricity_2_3 = symmetry_error(sw_cycles2_r[1:], st_cycles3_r)
        error_symmetricity_3_2 = symmetry_error(sw_cycles2_r, st_cycles3_r[:-1])

        error_symmetricity_1_4 = symmetry_error(sw_cycles1_l[1:], st_cycles4_l)
        error_symmetricity_4_1 = symmetry_error(sw_cycles4_l, st_cycles1_l[:-1])
        #############################################################################################################

        error_symmetricity1_front = error_symmetricity_1_2 + error_symmetricity_2_1
        error_symmetricity1_hind = error_symmetricity_4_3 + error_symmetricity_3_4

        error_symmetricity1_2_left = error_symmetricity_1_4 + error_symmetricity_4_1
        error_symmetricity1_2_right = error_symmetricity_2_3 + error_symmetricity_3_2

        error_symmetricity1 = error_symmetricity1_front + error_symmetricity1_hind + \
                              error_symmetricity1_2_left + error_symmetricity1_2_right

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        ########################
        swing3 = s3_state < 0
        stance3 = s3_state > 0

        swing4 = s4_state < 0
        stance4 = s4_state > 0
        ########################

        sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)

        ################################################
        sw2_in_st3 = np.sum(swing2 & stance3) / np.sum(swing2)
        sw3_in_st2 = np.sum(swing3 & stance2) / np.sum(swing3)

        sw1_in_st4 = np.sum(swing1 & stance4) / np.sum(swing1)
        sw4_in_st1 = np.sum(swing4 & stance1) / np.sum(swing4)

        sw3_in_st4 = np.sum(swing3 & stance4) / np.sum(swing3)
        sw4_in_st3 = np.sum(swing4 & stance3) / np.sum(swing4)

        sw1_in_sw4 = 1 - (np.sum(swing1 & swing4) / np.sum(swing4))
        sw2_in_sw3 = 1 - (np.sum(swing2 & swing3) / np.sum(swing3))

        ################################################

        error_symmetricity2 = (1 - sw1_in_st2) + (1 - sw2_in_st1) + (1 - sw3_in_st4) + (1 - sw4_in_st3) + \
                              (1 - sw2_in_st3) + (1 - sw3_in_st2) + (1 - sw1_in_st4) + (1 - sw4_in_st1) + \
                              (1 - sw1_in_sw4) + (1 - sw2_in_sw3)

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity1 = 10
        error_symmetricity2 = 10

    error = 2 * error_phase + error_speed + \
            error_symmetricity1 + 3 * error_symmetricity2

    return history, error, error_phase, error_speed, error_symmetricity1, error_symmetricity2


def only_error(history, time=95):
    eval_slice = int((time / 95) * len(history["s1_state"]))
    s1_state = history["s1_state"][:eval_slice]
    s2_state = history["s2_state"][:eval_slice]

    try:
        left_sw_cycles, left_st_cycles = calc_swing_stance(s1_state)
        right_sw_cycles, right_st_cycles = calc_swing_stance(s2_state)

        error_left_phase, error_left_speed = single_limb_error(left_sw_cycles, left_st_cycles)
        error_right_phase, error_right_speed = single_limb_error(right_sw_cycles, right_st_cycles)

        error_phase = error_left_phase + error_right_phase
        error_speed = 0

        error_symmetricity_l_r = symmetry_error(left_sw_cycles[1:], right_st_cycles)
        error_symmetricity_r_l = symmetry_error(right_sw_cycles, left_st_cycles[:-1])

        error_symmetricity1 = error_symmetricity_l_r + error_symmetricity_r_l

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)

        error_symmetricity2 = (1 - sw1_in_st2) + (1 - sw2_in_st1)

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity1 = 10
        error_symmetricity2 = 10

    error = 2 * error_phase + error_speed + \
            error_symmetricity1 + error_symmetricity2

    return error, error_phase, error_speed, error_symmetricity1, error_symmetricity2

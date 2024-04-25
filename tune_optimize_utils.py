from datetime import datetime
from ray import tune, train


def ray_wrapper(func):
    def inner(params):
        history, error, error_phase, error_speed, error_sym1, error_sym2 = func(params)
        train.report(dict(error=error,
                          error_phase=error_phase,
                          error_speed=error_speed,
                          error_symmetricity1=error_sym1,
                          error_symmetricity2=error_sym2
                          )
                     )

    return inner


def exp_name(Alg):
    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    return f"{Alg}{now}"


best_params = [
    {"init_stance": 0.50191,
     "init_stance_position": 0.49604,
     "init_swing": 4.6773,
     "speed_stance": 3.8701,
     "speed_swing": 3.4240,
     "inner_inhibit": -0.44423,
     "st_st_con": 0.77360,
     "st_sw_con": -0.95284,
     "sw_st_con": -0.95586,
     "sw_sw_con": 0.082690,
     'sw_sw_con_new': -0.457560934339573,
     'st_sw_con_new': -0.7718541665303388,
     'sw_st_con_new': -0.7451284086805224,
     'st_st_con_new': 0.048201438505927174}
]

# parameters search space for tune
search_space = {
    "init_stance": 0.50191,
    "init_stance_position": 0.49604,
    "init_swing": 4.6773,
    "speed_stance": 3.8701,
    "speed_swing": 3.4240,
    "inner_inhibit": -0.44423,
    "st_st_con": 0.77360,
    "st_sw_con": -0.95284,
    "sw_st_con": -0.95586,
    "sw_sw_con": 0.082690,
    "sw_sw_con_new": tune.uniform(-1, 1),
    "st_sw_con_new": tune.uniform(-1, 0),
    "sw_st_con_new": tune.uniform(-1, 0),
    "st_st_con_new": tune.uniform(0, 1)
}

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
     'sw_sw_con': 0.83093,
     'st_sw_con': -0.70609,
     'sw_st_con': -0.77690,
     'st_st_con': 0.48151,
     'sw_sw_con_new': -0.81191,
     'st_sw_con_new': -0.89918,
     'sw_st_con_new': -0.71148,
     'st_st_con_new': 0.04916}
]

# parameters search space for tune
search_space = {
    "init_stance": tune.uniform(0.5, 1.5),
    "init_stance_position": tune.uniform(0, 1),
    "init_swing": tune.uniform(4.5, 5.5),
    "speed_stance": tune.uniform(3, 4),
    "speed_swing": tune.uniform(2, 5),
    "inner_inhibit": tune.uniform(-1, 0),
    "sw_sw_con": tune.uniform(-1, 1),
    "st_sw_con": tune.uniform(-1, 0),
    "sw_st_con": tune.uniform(-1, 0),
    "st_st_con": tune.uniform(0, 1),
    "sw_sw_con_new": tune.uniform(-1, 1),
    "st_sw_con_new": tune.uniform(-1, 0),
    "sw_st_con_new": tune.uniform(-1, 0),
    "st_st_con_new": tune.uniform(0, 1)
}

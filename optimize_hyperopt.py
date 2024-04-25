import os
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from optimize import simulation_error
import tune_optimize_utils as utils


if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = '0'
    ray_simulation_error = utils.ray_wrapper(simulation_error)

    algo = HyperOptSearch(points_to_evaluate=utils.best_params)
    config = utils.search_space
    config["framework"] = "tf2"
    analysis = tune.run(
        ray_simulation_error,
        name=utils.exp_name("Hyperout"),
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=56,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=config)

    print("Best hyperparameters found were: ", analysis.best_config)

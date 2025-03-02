import optuna

from .objective_functions import build_objective_sae, build_objective_uae, build_objective_uvae

def run_optuna_study(direction: str, pruner, objective_function, n_trials: int) -> optuna.Study:
    study_uae = optuna.create_study(direction=direction, pruner=pruner)
    study_uae.optimize(objective_function, n_trials = n_trials)
    return study_uae

def save_optimal_parameters(study: optuna.Study, model_name: str, experiment_name: str):
    pass
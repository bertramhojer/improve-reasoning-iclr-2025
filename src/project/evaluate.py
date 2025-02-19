from repana import ControlModel, ControlVector, evaluate, eval_kld, eval_entropy, eval_prob_mass
import pickle
import os
import numpy as np
import random
from datetime import datetime
import polars as pl
import hydra
from omegaconf import DictConfig


def load_data(task: str = "babi", n: int = 1000, shots: int = 1):

    if task == "babi":
        data_path = os.path.join('data/bAbI', f"task_15_train_{shots}_shot.csv")
        data = pl.read_csv(data_path)
        data = data.sample(n=n)
        X, y = [x for x in data["question"]], [y for y in data["answer"]]
        return X, y
    
    data_path = os.path.join('data/IOI', f"{task}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    tmp = list(zip(data[shots]["X_test"], data[shots]["y_test"])) 
    return zip(*random.sample(tmp, k=n))  

def load_control_vector(path: str) -> ControlVector:
    return ControlVector.load(path)

def get_model_settings(model: ControlModel):
    return {
        "pad_token_id": model.tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": 10,
        "stop_strings": ["\n", "Passage:"],
        "tokenizer": model.tokenizer,
    }

def save_results(results, f: str):
    for col in results.columns:
        if results[col].dtype == object:
            results = results.with_columns(pl.col(col).cast(pl.Utf8))
    date = datetime.now().strftime("%Y%m%d")
    filename = f"{datetime.now().strftime('%H%M%S')}-{f}.csv"
    result_dir = os.path.join("results", date)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, filename)
    results.write_csv(result_path)
    print(f"Results saved to: {result_path}")


def run_evaluation(cfg):
    
    result_df = pl.DataFrame()
    layers = cfg.model.layer_id
    model = ControlModel(model_name=cfg.model.path, layer_ids=[layers])
    settings = get_model_settings(model)

    cv_path = f"cv/{cfg.params.eval_cv}/{cfg.params.cv_type}/{cfg.model.name}-{cfg.data.shots}.pkl"
    cv = load_control_vector(cv_path)

    i, j, k = cfg.params.alpha
    alphas = [round(i, 2) for i in np.arange(i, j+k, k)]

    X, y = load_data(task=cfg.data.task, n=cfg.params.samples, shots=cfg.data.test_shots)
    answer_list = list(set(y))

    for alpha in alphas:
        print(f"Evaluating {cfg.model.name} with alpha = {alpha}, CV trained on {cfg.data.task}")
        text_answers, accuracy = evaluate(
            model_type='pythia', model=model, control_vector=cv, alpha=alpha,
            normalize=cfg.params.normalize, X=X, y=y, type=cfg.params.eval_type, answer_list=answer_list,
            batch_size=cfg.params.batch_size, settings=settings
        )
        
        kld_mu, kld_std = eval_kld(
            model=model, control_vector=cv, normalize=cfg.params.normalize,
            alpha=alpha, X=X, y=y, settings=settings
        )

        entropy_mu, entropy_std = eval_entropy(
            model=model, control_vector=cv, normalize=cfg.params.normalize,
            alpha=alpha, X=X, y=y, settings=settings
        )

        mean_correct_prob, var_correct_prob, mean_incorrect_prob, var_incorrect_prob = eval_prob_mass(
            model_type='pythia', model=model, control_vector=cv,
            normalize=cfg.params.normalize, alpha=alpha, X=X, y=y,
            answer_list=answer_list, mask_to_answer_list=True, settings=settings
        )

        res_row = pl.DataFrame({
            "model": cfg.model.name, 
            "cv": cfg.params.cv_type, 
            "normalized": cfg.params.normalize, 
            "alpha": alpha,
            "accuracy": accuracy,
            "kld_mean": kld_mu,
            "kld_var": kld_std,
            "entropy_mean": entropy_mu, 
            "entropy_var": entropy_std,
            "correct_mean": mean_correct_prob,
            "correct_var": var_correct_prob,
            "incorrect_mean": mean_incorrect_prob, 
            "incorrect_var": var_incorrect_prob,
            "shots": cfg.data.shots, 
            "test_shots": cfg.data.test_shots, 
            "task": cfg.data.task,
            "eval_task": cfg.data.task
        })

        result_df = result_df.vstack(res_row)
    
    try:
        save_results(results=result_df, f=f"{cfg.model.name}_{cfg.data.shots}_{cfg.data.task}")
    except Exception as e:
        print("Couldn't save results \n", "Exception: " , e)
        
@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    cfg = config.experiment
    run_evaluation(cfg)
    
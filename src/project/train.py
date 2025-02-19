from repana import PCAContrastVector, ReadingContrastVector, ReadingVector, Dataset, evaluate
import os
import pickle
import polars as pl
import hydra
from omegaconf import DictConfig


def load_data(task, shots, model_type):
        
    if task == "babi":
        if model_type == 'mistral-instruct':
            data_path = os.path.join('data/bAbI/', 'new_balanced_rand.csv')
            data = pl.read_csv(data_path)
            return data
        data_path = os.path.join('data/bAbI/', f"task_15_train_{shots}_shot.csv")
        data = pl.read_csv(data_path)
        return data
    else:
        data_path = os.path.join('data/IOI/', f"{task}.pkl")
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data[shots]

def create_dataset(task, data, model_type, cv_type) -> Dataset:

    if task == 'babi':

        if model_type == 'mistral-instruct':
            positive = [f'{x}{y}' for x, y in zip(data['question'], data['answer'])]
        else:
            positive = [f"{x} {y}" for x, y in zip(data["question"], data['answer'])]
        if cv_type != "R":
            negative = [f"{x}" for x in data["rand"]]
        else: negative = None
        return Dataset(positive=positive, negative=negative)

    else:
        positive = [f"{x} {y}" for x, y in zip(data["X_train"], data["y_train"])]
        if cv_type == "R":
            negative = None
        else:
            negative = data["X_train_negative"]
        return Dataset(positive=positive, negative=negative)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:

    cfg = config.experiment

    print(f"Training CV's for {cfg.model.name} on task {cfg.data.task}")
    data = load_data(cfg.data.task, cfg.data.shots, cfg.model.model_type)
    dataset = create_dataset(cfg.data.task, data, cfg.model.model_type, cfg.params.cv_type)

    cv_class = {
        "R": ReadingVector,
        "R-C": ReadingContrastVector,
        "PCA-C": PCAContrastVector
    }.get(cfg.params.cv_type)
    
    if cv_class:
        cv = cv_class(model_name=cfg.model.path, standardize=cfg.params.standardize)
        cv.train(dataset)
        cv.save(cfg.data.task, cfg.params.cv_type, cfg.data.shots)
    else:
        print(f"Unsupported CV type: {cfg.params.cv_type}")

if __name__ == "__main__":
    main()

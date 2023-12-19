import torch
import datetime
from pathlib import Path

class ModelSaveHandler:
    def __init__(self, date_string, model_dir, min_delta=0.5):
        self.counter = 0
        self.min_validation_loss = float('inf')

        date_string = date_string.replace(' ', "_").replace('-', "_").replace(':', "_")
        self.current_model_dir = f"{model_dir}/{date_string}"
        Path(self.current_model_dir).mkdir(parents=True, exist_ok=True)
        self.best_model_dir = f"{self.current_model_dir}/bestmodels"
        Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)

        self.min_delta = min_delta
        self.saved_model_paths = []

    def save_model(self, validation_loss, model):
        if validation_loss < self.min_validation_loss + self.min_delta:
            self.min_validation_loss = validation_loss
            model_path = f'{self.current_model_dir}/{self.counter}.pt'
            torch.save(model.state_dict(), model_path)
            self.saved_model_paths.append(model_path)
            self.counter += 1

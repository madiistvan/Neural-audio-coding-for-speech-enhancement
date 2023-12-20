import torch
from EncodedDataset import EncodedDataset
from torch.utils.data import DataLoader, random_split
from LatentNetwork import LatentNetwork
import torch.optim as optim

from clearml import Task
import datetime
from ModelSaveHandler import ModelSaveHandler

class LatentTrainer:
    def __init__(self, model_dir, data_dir, noise_dir):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.noise_dir = noise_dir

    def create_dataset(
        self,
        data_dir,
        noise_dir,
        batch_size = 42,
    ):
        
        dataset = EncodedDataset(data_dir, noise_dir)

        total_samples = len(dataset)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        test_size = total_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def create_task_and_logger(self, device, project_name="speech-enhancement"):
        print(f"Device: {device}")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
        task_name = "Latent space test"
        date_string = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        task_name += " " + date_string
        task = Task.init(project_name=project_name, task_name=task_name)
        logger = task.get_logger()
        return task, logger

    def training_loop(self, device, model, train_loader, val_loader, logger, optimizer, NUM_EPOCHS, model_save_handler, test_loader):
        criterion = torch.nn.MSELoss()

        for epoch in range(NUM_EPOCHS):
            model.train()
            losses = 0.0
            for noisy, target in train_loader:
                noisy = noisy.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                
                pred = model(noisy.squeeze(1))
                loss = criterion(pred, target.squeeze(1))
                losses += loss
                loss.backward()
                optimizer.step()

            avg_loss = losses = len(train_loader)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')
            self._log_scalar(logger, avg_loss, epoch)

            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for noisy, target in val_loader:
                    noisy = noisy.to(device)
                    target = target.to(device)
                    val_pred = model(noisy.squeeze(1))
                    target = target.squeeze(1)
                    val_loss += criterion(val_pred, target)

                average_val_loss = val_loss / len(val_loader)
                print(f'Validation Loss: {average_val_loss:.4f}')
                self._log_scalar(logger, average_val_loss, epoch, data_name = "Latent Val loss")
                model_save_handler.save_model(average_val_loss, model)

        with torch.no_grad():
            model.eval()
            for i_test, j_test in test_loader:
                test_pred = model(i_test.squeeze(1))
                j_test = j_test.squeeze(1)
                test_loss = criterion(test_pred, j_test)
            print(f'Test Loss: {test_loss.item():.4f}')

    def _log_scalar(self, logger, loss, epoch, data_name = "Latent Loss", title = "Loss"):
        logger.report_scalar(title=title, series=data_name, value=loss, iteration=epoch)

    def train(self, num_epochs = 100, batch_size = 42):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            date_string = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            model_save_handler = ModelSaveHandler(date_string, model_dir=self.model_dir)

            model = LatentNetwork()
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001)

            train_loader, val_loader, test_loader = self.create_dataset(
                data_dir = self.data_dir,
                noise_dir = self.noise_dir,
                batch_size=batch_size)
            task, logger = self.create_task_and_logger(device)
            model.to(device)

            self.training_loop(device, model, train_loader, val_loader, logger, optimizer, num_epochs, model_save_handler, test_loader)
        
            torch.save(model.state_dict(), f'./saved_models/{date_string}')
        finally:
            task.close()

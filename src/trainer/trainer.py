import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

class Trainer:

    def __init__(self,
        models: dict[str, torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
        criterions: list[torch.nn.Module],
        schedulers: list[torch.optim.lr_scheduler._LRScheduler],
        log_dir: str = "logs"
        ):

        self.models = models
        self.criterions = criterions
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.writers = [
            SummaryWriter(log_dir=f"{log_dir}/{model_name}")
            for model_name in models.keys()
        ]

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            torch.backends.cudnn.benchmark = True
        
    def train(self,
        num_epochs: int = 350,
        train_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        ) -> None:

        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        i = 0
        for model_name, model in self.models.items():
            print(f"Training model : {model_name}")

            model.to(self.device)

            epoch_bar = tqdm(range(num_epochs), desc="Training Epochs")

            for epoch in epoch_bar:

                total_loss, correct, total = 0, 0, 0

                for inputs, targets in train_loader:

                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizers[i].zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        outputs = model(inputs)
                        loss = self.criterions[i](outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizers[i])
                    scaler.update()
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()


                train_acc = correct / total
                train_loss = total_loss / total
                self.writers[i].add_scalar('Train/Loss', train_loss, epoch)
                self.writers[i].add_scalar('Train/Accuracy', train_acc, epoch)

                model.eval()

                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                            outputs = model(inputs)
                            loss = self.criterions[i](outputs, targets)
                        if torch.isnan(loss):
                            print(f"[NaN Detected] Model: {list(self.models.keys())[i]}, Epoch: {epoch}")
                            print("Loss is NaN. Skipping this batch.")
                            continue
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                val_acc = val_correct / val_total
                val_loss = val_loss / val_total
                self.writers[i].add_scalar('Val/Loss', val_loss / val_total, epoch)
                self.writers[i].add_scalar('Val/Accuracy', val_acc, epoch)

                self.schedulers[i].step()
                epoch_bar.set_postfix({
                    "Train Acc": f"{train_acc:.4f}",
                    "Val Acc": f"{val_acc:.4f}",
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}"
                })

            # Clean gpu if using it:
            if self.device ==  "cuda":
                torch.cuda.empty_cache()
            i += 1

    def evaluate(
            self,
            test_loader: torch.utils.data.DataLoader,
        ) -> dict[str, float]:

        i = 0
        test_accuracies = {}
        for model_name, model in self.models.items():

            model.to(self.device)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = correct / total
            self.writers[i].add_scalar('Test/Accuracy', test_acc)
            self.writers[i].close()

            i += 1

            test_accuracies[model_name] = test_acc

        return test_accuracies
        
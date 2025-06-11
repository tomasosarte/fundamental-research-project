import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

torch.set_float32_matmul_precision("medium")

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

        self.writers = [SummaryWriter(log_dir=f"{log_dir}/{name}")
                        for name in models.keys()]

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
    def train(self,
        num_epochs: int = 350,
        train_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        ) -> None:

        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        for (model_name, model), optimizer, criterion, scheduler, writer in zip(
            self.models.items(), self.optimizers, self.criterions, self.schedulers, self.writers
        ):
            print(f"Training model : {model_name}")

            model.to(self.device)

            epoch_bar = tqdm(range(num_epochs), desc="Training Epochs")

            for epoch in epoch_bar:
                
                model.train()
                total_loss, correct, total = 0, 0, 0

                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()


                train_acc = correct / total
                train_loss = total_loss / total
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('Train/Accuracy', train_acc, epoch)

                model.eval()

                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        if torch.isnan(loss):
                            print(f"[NaN Detected] Model: {model_name}, Epoch: {epoch}")
                            continue
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                if val_total > 0:
                    val_acc = val_correct / val_total
                    val_loss = val_loss / val_total
                else:
                    print(
                        f"No valid validation samples for model {model_name} at epoch {epoch}."
                    )
                    val_acc = float('nan')
                    val_loss = float('nan')

                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Accuracy', val_acc, epoch)

                scheduler.step()
                epoch_bar.set_postfix({
                    "Train Acc": f"{train_acc:.4f}",
                    "Val Acc": f"{val_acc:.4f}",
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}"
                })

            # Clean gpu if using it:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def evaluate(
            self,
            test_loader: torch.utils.data.DataLoader,
        ) -> dict[str, float]:

        test_accuracies = {}
        for (model_name, model), writer in zip(self.models.items(), self.writers):

            model.to(self.device)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = correct / total
            writer.add_scalar('Test/Accuracy', test_acc)
            writer.close()
            test_accuracies[model_name] = test_acc

        return test_accuracies
        
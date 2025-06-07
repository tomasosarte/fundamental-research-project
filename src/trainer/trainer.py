import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        ):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=f"logs/{model.__class__.__name__}")
        
    def train(self, 
        num_epocs: int= 350,
        train_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        ):
        
        for epoch in range(num_epocs):

            self.model.train()
            total_loss, correct, total = 0, 0, 0


            for inputs, targets in train_loader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


            train_acc = correct / total
            self.writer.add_scalar('Train/Loss', total_loss / total, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)

            self.model.eval()

            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            val_acc = val_correct / val_total
            self.writer.add_scalar('Val/Loss', val_loss / val_total, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

            scheduler.step()
            print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    def evaluate(
            self,
            test_loader: torch.utils.data.DataLoader,
        ):
    
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = correct / total
        self.writer.add_scalar('Test/Accuracy', test_acc)
        self.writer.close()
        print(f"\n Final Test Accuracy: {test_acc:.4f}")
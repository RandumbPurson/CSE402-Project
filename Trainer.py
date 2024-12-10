import numpy as np

from DictList import DictList


class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler,
            device="cpu",
            optimizer_kwargs={}
        ):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion()
        self.optimizer = optimizer(
            self.model.parameters(), **optimizer_kwargs
        )
        self.scheduler = scheduler(self.optimizer, [5, 10])  

    def step(
        self, loader, data_hook = lambda d, bd: None
    ):
        data = DictList()
        for i, (x, y) in enumerate(loader):
            bdata = {"x": x, "y": y}
            bdata["y_probs"] = self.model(bdata["x"].to(self.device))
            bdata["loss_grad"] = self.criterion(
                bdata["y_probs"], bdata["y"].to(self.device)
            )
            
            if self.model.training:
                bdata["loss_grad"].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            data_hook(data, bdata) # Run optional data hook

        if self.model.training:
            self.scheduler.step()

        return data

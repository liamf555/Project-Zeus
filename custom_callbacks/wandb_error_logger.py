import wandb
from wandb.integration.sb3 import WandbCallback

class WandbErrorLogger(WandbCallback):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
import hydra
import torch
from omegaconf import DictConfig
from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping


import sys
sys.path.append('./models')
from dataset import GlaucomaLDM
from models import LightningModel

"""
Training script to test each model.
"""

@hydra.main(version_base=None, config_name="config", config_path="configs")
def train(cfg: DictConfig):
    # connect to ClearML
    task = Task.init(project_name="Glaucoma Classification",
                     task_name=f"Train Model - {cfg['model']}")

    # initialize data module and model
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(device)
    print("   \   ")
    print("   ^ ^ ")
    print("  {'.'}")
    print("? /  | ")
    print("\|  || ")

    torch.backends.cudnn.benchmark = True if torch.backends.cudnn.is_available() and cuda_available else False
    torch.set_float32_matmul_precision('high')

    datamodule = GlaucomaLDM(batch_size=1, num_workers=0)

    model = LightningModel(model=cfg['model'], clearml_logger=task.get_logger()).to(device)

    # initialize checkpoint callback depending on parse arguments
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints_{cfg['model']}/", monitor="val_loss", mode="min")

    trainer = Trainer(accelerator="gpu", max_epochs=1, profiler="simple",
                          callbacks=[checkpoint_callback, ModelSummary(4),
                                     EarlyStopping(monitor="val_loss",
                                                   mode="min",
                                                   patience=10)],
                          strategy="auto", enable_checkpointing=True)

    trainer.fit(model=model, datamodule=datamodule)
    
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    train()
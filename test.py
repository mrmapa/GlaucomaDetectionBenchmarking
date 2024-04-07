import hydra
import torch
from omegaconf import DictConfig
from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping


import sys
sys.path.append('./models')
from dataset import GlaucomaLDM, GlaucomaSegmentationLDM
from models import LightningModel

"""
Training script to test each model.
"""

@hydra.main(version_base=None, config_name="config_test", config_path="configs")
def test(cfg: DictConfig):
    # connect to ClearML
    task = Task.init(project_name="Glaucoma Classification",
                     task_name=f"Test Model - {cfg['model']}")

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

    if cfg['model'] == 'FCDense':
        model = LightningModel.load_from_checkpoint("./checkpoints_FCDense_classification/epoch=26-step=34776.ckpt", model=cfg['model'],
                                                    mode="classification", clearml_logger=task.get_logger()).to(device)
        model.model.mode = "classification"

    elif cfg['model'] == 'Early':
        model = LightningModel.load_from_checkpoint('./checkpoints_Early/epoch=52-step=102343.ckpt', model=cfg['model'],
                                                    mode='classification', clearml_logger=task.get_logger()).to(device)
    
    elif cfg['model'] == 'U-Net':
        model = LightningModel.load_from_checkpoint('./checkpoints_U-Net_classification/epoch=21-step=28336.ckpt', model=cfg['model'],
                                                    mode='classification', clearml_logger=task.get_logger()).to(device)
        model.model.mode = "classification"
        
    trainer = Trainer(accelerator="gpu", max_epochs=1, profiler="simple",
                callbacks=[ModelSummary(4),
                            EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=10)],
                strategy="auto", enable_checkpointing=True)
    datamodule = GlaucomaLDM(batch_size=6, num_workers=4)
    trainer.test(model=model, datamodule=datamodule)   

    return trainer.callback_metrics["test_accuracy"].item(), trainer.callback_metrics["test_precision"].item(), trainer.callback_metrics["test_recall"].item(), trainer.callback_metrics["test_f1"].item()
    
    if cfg['model'] == 'U-Net':
        datamodule = GlaucomaSegmentationLDM(batch_size=6, num_workers=4)

        model = LightningModel(model=cfg['model'], mode="segmentation").to(device) # set mode to classification if training classification-only model

        # initialize checkpoint callback depending on parse arguments
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints_{cfg['model']}_segmentation/", monitor="val_loss", mode="min")

        trainer = Trainer(accelerator="gpu", max_epochs=100, profiler="simple",
                            callbacks=[checkpoint_callback, ModelSummary(4),
                                        EarlyStopping(monitor="val_loss",
                                                    mode="min",
                                                    patience=10)],
                            strategy="auto", enable_checkpointing=True)

        trainer.fit(model=model, datamodule=datamodule)

        # switch from segmentation to classification, comment out if not classifying
        model.mode = "classification"
        model.model.mode = "classification"

        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints_{cfg['model']}_classification/", monitor="val_loss", mode="min")

        trainer = Trainer(accelerator="gpu", max_epochs=100, profiler="simple",
                            callbacks=[checkpoint_callback, ModelSummary(4),
                                        EarlyStopping(monitor="val_loss",
                                                    mode="min",
                                                    patience=10)],
                        strategy="auto", enable_checkpointing=True)
        
        datamodule = GlaucomaLDM(batch_size=6, num_workers=4)
        trainer.fit(model=model, datamodule=datamodule)
    
        return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    test()
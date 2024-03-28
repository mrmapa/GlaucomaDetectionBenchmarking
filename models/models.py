from torch import nn
import pytorch_lightning as pl
import torch
import sys
sys.path.append('./models/Optic-Disk-Cup-Segmentation/src/segmentation code')
from model import FCDenseNet57
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics import MetricCollection


class conv_block(nn.Module):
    """
    Convolutional block consisting of:
      1. 3 x 3 convolutional layer from in_channels to out_channels
      2. ReLU()
      3. Dropout() w/ p = 0.5
      4. 3 x 3 convolutional layer from out_channels to out_channels
      5. ReLU()
      6. Dropout() w/ p = 0.5

    Args:
        in_channels - input channels
        out_channels - output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # initialize layers of the convolutional block
        self.conv1 = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        # pass tensors through the layers of the model
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class encoder_block(nn.Module):
    """
    Encoder block consisting of convolutional block as defined above + max pooling layer
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.maxpool(skip)
        return x, skip

class decoder_block(nn.Module):
    """
    Decoder block consisting of upsampling, concatenation and convolutional block
    """
    def __init__(self, in_channels: int, up_channels: int, skip_channels: int, out_channels: int, kernel_size=2, out_padding=0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=kernel_size, stride=2, padding=0, output_padding=out_padding)
        self.conv = conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        combined = torch.cat((x, skip), axis=1)
        combined = self.conv(combined)
        return combined


class U_Net(nn.Module):
    """
    U-Net implementation based on the paper "Optic Disc and Cup Segmentation Methods for Glaucoma Detection
      With Modification of U-Net Convolutional Neural Network." This network has been modified for the task
      of glaucoma classification by adding a classification head to the output of the network after passing
      through the U-Net layers.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # initialize encoder, decoder layers of the U-Net model
        self.encoder_1 = encoder_block(in_channels, 32)
        self.encoder_2 = encoder_block(32, 64)
        self.encoder_3 = encoder_block(64, 64)
        self.encoder_4 = encoder_block(64, 64)
        self.bottleneck = conv_block(64, 64)

        self.decoder_1 = decoder_block(64, 64, 64, 64)
        self.decoder_2 = decoder_block(64, 64, 64, 64)
        self.decoder_3 = decoder_block(64, 64, 64, 64)
        self.decoder_4 = decoder_block(64, 64, 32, 32, 1, 1)

        self.conv = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.dropout2d = nn.Dropout2d(p=0.3)

        # initialize classifier head
        self.fcn1 = nn.Linear(512*512, 1024)
        self.fcn2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # pass image through encoder and decoder layers
        x, skip_1 = self.encoder_1(x)
        x, skip_2 = self.encoder_2(x)
        x, skip_3 = self.encoder_3(x)
        x, skip_4 = self.encoder_4(x)
        x = self.bottleneck(x)
        x = self.decoder_1(x, skip_4)
        x = self.decoder_2(x, skip_3)
        x = self.decoder_3(x, skip_2)
        x = self.decoder_4(x, skip_1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = self.dropout2d(x)

        # flatten image tensor and pass through classifier head layers
        x = torch.flatten(x, start_dim=1)
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fcn2(x)
        x = self.sigmoid(x)

        return x

class FCDenseModified(nn.Module):
    """
    Model based on the paper "Enhanced Optic Disk and Cup Segmentation with Glaucoma Screening
    from Fundus Images using Position encoded CNNs". This model combines a segmentation network
    with a classification head to differentiate between healthy and glaucoma images.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.model = FCDenseNet57(1, inchannels=in_channels)

        # initialize classifier head
        self.fcn1 = nn.Linear(512*512, 1024)
        self.fcn2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pass image through FCDenseNet
        x = self.model(x)

        # flatten image and pass through classifier head layers
        x = torch.flatten(x, start_dim=1)
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fcn2(x)
        x = self.sigmoid(x)

        return x

class conv_relu_batchnorm_block(nn.Module):
    """
    Block containing convolutional layer, ReLU, batch normalization, and optional maxpooling layer.
    """
    def __init__(self, in_channels: int, out_channels: int, maxpool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

        if maxpool:
            self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        
        if self.maxpool:
            x = self.maxpool(x)

        x = self.batch_norm(x)

        return x

class EarlyId(nn.Module):
    """
    Model based on the paper "A Deep Learning Approach to Automatic Detection of Early Glaucoma
    from Visual Fields". This model implements the model as-is but using PyTorch.
    """
    def __init__(self, in_channels: int = 3):
        # define convolutional layers
        super().__init__()
        self.conv1 = conv_relu_batchnorm_block(3, 4, False)
        self.conv2 = conv_relu_batchnorm_block(4, 4, False)
        self.conv3 = conv_relu_batchnorm_block(4, 4, True)
        self.conv4 = conv_relu_batchnorm_block(4, 4, False)
        self.conv5 = conv_relu_batchnorm_block(4, 4, False)

        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.avgpool = nn.AvgPool2d(stride=2, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(4)

        # define fully connected layers
        self.fc1 = nn.Linear(4*64*64, 32)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.final = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pass image through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # pass image through fully connected layers
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = self.bn1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x

class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning model for testing out each of the models.
    """
    def __init__(self, model: str, clearml_logger = None):
        super().__init__()
        if model == "U-Net":
            self.model = U_Net()
        elif model == "FCDense":
            self.model = FCDense()
        elif model == "Early":
            self.model = EarlyId()

        self.loss_fn = nn.BCELoss()

        self.clearml_logger = clearml_logger

        self.train_accuracy = BinaryAccuracy()
        self.train_metrics = MetricCollection({
            'train_precision': BinaryPrecision(),
            'train_recall': BinaryRecall(),
            'train_f1': BinaryF1Score()
        })

        self.val_accuracy = BinaryAccuracy()
        self.val_metrics = MetricCollection({
            'val_precision': BinaryPrecision(),
            'val_recall': BinaryRecall(),
            'val_f1': BinaryF1Score()
        })

        self.test_accuracy = BinaryAccuracy()
        self.test_metrics = MetricCollection({
            'test_precision': BinaryPrecision(),
            'test_recall': BinaryRecall(),
            'test_f1': BinaryF1Score()
        })

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        y.resize_(outputs.size(dim=0), 1)

        loss = self.loss_fn(outputs, y.to(torch.float))

        self.train_accuracy.update(outputs, y)

        if torch.any(y == 1):
            self.train_metrics.update(outputs, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
        
    def on_training_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        accuracy = self.train_accuracy.compute()
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        y.resize_(outputs.size(dim=0), 1)

        loss = self.loss_fn(outputs, y.to(torch.float))

        self.val_accuracy.update(outputs, y)

        if torch.any(y == 1):
            self.val_metrics.update(outputs, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

        accuracy = self.val_accuracy.compute()
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics.reset()
        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        y.resize_(outputs.size(dim=0), 1)

        loss = self.loss_fn(outputs, y.to(torch.float))

        self.test_accuracy.update(outputs, y)

        if torch.any(y == 1):
            self.test_metrics.update(outputs, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)

        accuracy = self.test_accuracy.compute()
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics.reset()
        self.test_accuracy.reset()
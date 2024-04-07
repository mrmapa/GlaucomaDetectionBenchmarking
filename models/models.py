from torch import nn
import pytorch_lightning as pl
import torch
import sys
sys.path.append('./models/Optic-Disk-Cup-Segmentation/src/segmentation code')
from model import FCDenseNet57
from torchmetrics.classification import BinaryConfusionMatrix, BinaryJaccardIndex
from transformers import AutoImageProcessor, Swinv2ForImageClassification


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

class TestModel(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.fcn1 = nn.Linear(3*256*256, 32)
        self.fcn2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # flatten image tensor and pass through classifier head layers
        x = torch.flatten(x, start_dim=1)
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fcn2(x)
        x = self.sigmoid(x)

        return x

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
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        self.densenet.classifier = nn.Sequential(nn.Linear(1920, 1000, bias=True), nn.Dropout(p=0.5), nn.Linear(in_features=1000, out_features=1, bias=True))
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.train()

        self.mode = "segmentation"

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

        if self.mode == "segmentation":
            x = self.dropout2d(x)
            x = self.sigmoid(x)
            return x

        # flatten image tensor and pass through classifier head
        x = self.densenet(x)
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
        self.sigmoid = nn.Sigmoid()

        # initialize classifier head
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        self.densenet.classifier = nn.Sequential(nn.Linear(1920, 1000, bias=True), nn.Dropout(p=0.5), nn.Linear(in_features=1000, out_features=1, bias=True))
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.train()

        self.mode = "segmentation"

    def forward(self, x):
        # pass image through FCDenseNet
        x = self.model(x)

        if self.mode == "segmentation":
            x = self.sigmoid(x)
            return x

        # flatten image tensor and pass through classifier head
        x = self.densenet(x)
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
        self.conv1 = conv_relu_batchnorm_block(in_channels, 4, False)
        self.conv2 = conv_relu_batchnorm_block(4, 4, False)
        self.conv3 = conv_relu_batchnorm_block(4, 4, True)
        self.conv4 = conv_relu_batchnorm_block(4, 4, False)
        self.conv5 = conv_relu_batchnorm_block(4, 4, False)

        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.avgpool = nn.AvgPool2d(stride=2, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(4)

        # define fully connected layers
        self.fc1 = nn.Linear(4*32*32, 32)
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
    def __init__(self, model: str, clearml_logger = None, mode = "classification"):
        super().__init__()
        self.processor = None
        if model == "U-Net":
            self.model = U_Net()
        elif model == "FCDense":
            self.model = FCDenseModified()
        elif model == "Early":
            self.model = EarlyId()
        else:
            self.processor = AutoImageProcessor.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")
            self.model = Swinv2ForImageClassification.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")

        self.loss_fn = nn.BCELoss()

        self.clearml_logger = clearml_logger

        self.train_cm = BinaryConfusionMatrix()
        self.train_jaccard = BinaryJaccardIndex()

        self.val_cm = BinaryConfusionMatrix()
        self.val_jaccard = BinaryJaccardIndex()

        self.test_cm = BinaryConfusionMatrix()
        self.test_jaccard = BinaryJaccardIndex()

        self.mode = mode

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def on_train_epoch_start(self) -> None:
        self.model.train()
        return super().on_train_epoch_start()
    
    def on_validation_epoch_start(self) -> None:
        self.model.eval()
        return super().on_validation_epoch_start()
    
    def on_test_epoch_start(self) -> None:
        self.model.eval()
        return super().on_test_epoch_start()
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # if self.processor:
        #     x = self.processor(x, return_tensors="pt")
        outputs = self.forward(x)
        if self.mode == "classification":
            y.resize_(outputs.size(dim=0), 1)
        elif self.mode == "segmentation":
            y.resize_(outputs.size(dim=0), 1, 256, 256)

        loss = self.loss_fn(outputs.to(torch.float), y.to(torch.float))
        # loss.requires_grad = True

        if self.mode == "classification":
            self.train_cm.update(outputs, y)
        elif self.mode == "segmentation":
            self.train_jaccard.update(outputs, y)

        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
        
    def on_training_epoch_end(self):
        if self.mode == "classification":
            epoch_train_cm = self.train_cm.compute()

            accuracy = (epoch_train_cm[0][0] + epoch_train_cm[1][1]) / (epoch_train_cm[0][0] + epoch_train_cm[0][1] + epoch_train_cm[1][0] + epoch_train_cm[1][1])
            precision = (epoch_train_cm[1][1]) / (epoch_train_cm[1][1] + epoch_train_cm[0][1])
            recall = (epoch_train_cm[1][1]) / (epoch_train_cm[1][1] + epoch_train_cm[1][0])
            f1 = 2 * ((precision * recall) / (precision + recall))

            self.log_dict({
                'train_accuracy': accuracy,
                'train_precision': precision,
                'train_recall': recall,
                'train_f1': f1
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger:
                self.clearml_logger.report_scalar(
                    title='train_accuracy', series='train_accuracy', value=accuracy, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='train_precision', series='train_precision', value=precision, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='train_recall', series='train_recall', value=recall, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='train_f1', series='train_f1', value=f1, iteration=self.global_step
                )
            
            self.train_cm.reset()

        elif self.mode == "segmentation":
            jaccard = self.train_jaccard.compute()
            print(jaccard)

            self.log('train_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger:
                self.clearml_logger.report_scalar(
                    title='train_jaccard', series='train_jaccard', value=jaccard, iteration=self.global_step
                )
            
            self.train_jaccard.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # if self.processor:
        #     x = self.processor(x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.forward(x)
        if self.mode == "classification":
            y.resize_(outputs.size(dim=0), 1)
        elif self.mode == "segmentation":
            y.resize_(outputs.size(dim=0), 1, 256, 256)

        loss = self.loss_fn(outputs.to(torch.float), y.to(torch.float))

        if self.mode == "classification":
            self.val_cm.update(outputs, y)
        elif self.mode == "segmentation":
            self.val_jaccard.update(outputs, y)

        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def on_validation_epoch_end(self):
        if self.mode == "classification":
            epoch_val_cm = self.val_cm.compute()

            accuracy = (epoch_val_cm[0][0] + epoch_val_cm[1][1]) / (epoch_val_cm[0][0] + epoch_val_cm[0][1] + epoch_val_cm[1][0] + epoch_val_cm[1][1])
            precision = (epoch_val_cm[1][1]) / (epoch_val_cm[1][1] + epoch_val_cm[0][1])
            recall = (epoch_val_cm[1][1]) / (epoch_val_cm[1][1] + epoch_val_cm[1][0])
            f1 = 2 * ((precision * recall) / (precision + recall))

            self.log_dict({
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger and self.global_step > 0:
                self.clearml_logger.report_scalar(
                    title='val_accuracy', series='val_accuracy', value=accuracy, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='val_precision', series='val_precision', value=precision, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='val_recall', series='val_recall', value=recall, iteration=self.global_step
                )
                self.clearml_logger.report_scalar(
                    title='val_f1', series='val_f1', value=f1, iteration=self.global_step
                )
            
            self.val_cm.reset()

        elif self.mode == "segmentation":
            jaccard = self.val_jaccard.compute()

            self.log('val_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger and self.global_step > 0:
                self.clearml_logger.report_scalar(
                    title='val_jaccard', series='val_jaccard', value=jaccard, iteration=self.global_step
                )
            
            self.val_jaccard.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch

        # if self.processor:
        #     x = self.processor(x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.forward(x)
        if self.mode == "classification":
            y.resize_(outputs.size(dim=0), 1)
        elif self.mode == "segmentation":
            y.resize_(outputs.size(dim=0), 1, 256, 256)

        loss = self.loss_fn(outputs.to(torch.float), y.to(torch.float))

        if self.mode == "classification":
            self.test_cm.update(outputs, y)
        elif self.mode == "segmentation":
            self.test_jaccard.update(outputs, y)

        self.log('test_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def on_test_epoch_end(self):
        if self.mode == "classification":
            epoch_test_cm = self.test_cm.compute()

            accuracy = (epoch_test_cm[0][0] + epoch_test_cm[1][1]) / (epoch_test_cm[0][0] + epoch_test_cm[0][1] + epoch_test_cm[1][0] + epoch_test_cm[1][1])
            precision = (epoch_test_cm[1][1]) / (epoch_test_cm[1][1] + epoch_test_cm[0][1])
            recall = (epoch_test_cm[1][1]) / (epoch_test_cm[1][1] + epoch_test_cm[1][0])
            f1 = 2 * ((precision * recall) / (precision + recall))

            self.log_dict({
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger:
                self.clearml_logger.report_single_value(
                    'test_accuracy', value=accuracy
                )
                self.clearml_logger.report_single_value(
                    'test_precision', value=precision
                )
                self.clearml_logger.report_single_value(
                    'test_recall', value=recall
                )
                self.clearml_logger.report_single_value(
                    'test_f1', value=f1
                )
            
            self.test_cm.reset()

        elif self.mode == "segmentation":
            jaccard = self.test_jaccard.compute()

            self.log('test_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if self.clearml_logger and self.global_step > 0:
                self.clearml_logger.report_single_value(
                    title='test_jaccard', series='test_jaccard', value=jaccard, iteration=self.global_step
                )
            
            self.test_jaccard.reset()
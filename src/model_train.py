import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class MitochondrialEMDataset(Dataset):
    def __init__(self, images, masks):
        super().__init__()
        self.images = images
        self.masks = masks

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx].copy() / 255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.masks[idx].copy() / 255.0, dtype=torch.float32).unsqueeze(0)

        return image, mask


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final(dec1)

def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()


def train_an_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_total = 0
    train_iou = 0
    num_batches = 0

    for imgs, masks in tqdm(loader, desc='Train'):
        # push to device
        imgs, masks = imgs.to(device), masks.to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        train_iou += compute_iou(outputs, masks)

        # backward
        loss.backward()
        optimizer.step()

        # calc metrics
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_loss += loss.item() * imgs.size(
            0)  # scale loss to all images instead of avg. handles different batch sizes esp. for the last batch
        train_acc += (preds == masks).sum().item()
        train_total += masks.numel()
        num_batches += 1

    return train_loss / len(loader.dataset), train_acc / train_total, train_iou / num_batches


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_total = 0
    val_iou = 0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validation'):
            # push to device
            imgs, masks = imgs.to(device), masks.to(device)

            # get preds and loss
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_iou += compute_iou(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            val_loss += loss.item()
            val_acc += (preds == masks).sum().item()
            val_total += masks.numel()
            num_batches += 1

    return val_loss / len(loader.dataset), val_acc / val_total, val_iou / num_batches


def test_model(model, loader, logger, examples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    test_iou = 0
    num_batchs = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Testing'):
            # push to device
            imgs, masks = imgs.to(device), masks.to(device)

            # get preds and loss
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            test_iou += compute_iou(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            test_loss += loss.item()
            test_acc += (preds == masks).sum().item()
            test_total += masks.numel()
            num_batchs += 1

        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc/test_total:.4f}, Test IoU: {test_iou/num_batchs:.4f}")
        logger.info(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc/test_total:.4f}, Test IoU: {test_iou/num_batchs:.4f}")

        fig, axs = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
        for i, (img, mask) in enumerate(examples):
            img, mask = img.to(device), mask.to(device)
            output = model(img.unsqueeze(0))
            pred_mask = (torch.sigmoid(output) > 0.5).float()

            img = img.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            pred_mask = pred_mask.squeeze().cpu().numpy()

            axs[i, 0].imshow(img, cmap='gray')
            axs[i, 1].imshow(mask, cmap='gray')
            axs[i, 2].imshow(pred_mask, cmap='gray')

            axs[i, 0].set_title('Image')
            axs[i, 1].set_title('Mask')
            axs[i, 2].set_title('Prediction')

            for j in range(3):
                axs[i, j].axis('off')

        plt.tight_layout()
        plt.savefig('images/testing_examples.png', bbox_inches='tight')
        plt.close()


def train_model(model, train_loader, num_epochs, logger, val_loader=None):
    # set device, optimizer, loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    logger.info(f'Using device: {device}')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # track training
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_iou': [],
        'val_loss': [],
        'val_acc': [],
        'val_iou': []
    }

    model = model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc, train_iou = train_an_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)

        if val_loader:
            val_loss, val_acc, val_iou = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_iou'].append(val_iou)
            print(f"Epoch {epoch + 1}/{num_epochs}: ")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train IoU: {train_iou:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: ")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test IoU: {train_iou:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}: ")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train IoU: {train_iou:.4f}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: ")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train IoU: {train_iou:.4f}")

    return history

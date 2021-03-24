import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 256

def main():


   # dataset path
    batch3_train_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "batch_3", "train")
    batch3_valid_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "batch_3", "val")
    batch3_test_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "batch_3", "test")

    batch2_train_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "batch_2", "train")
    batch2_valid_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "batch_2", "val")
    batch2_test_dir = os.path.join(BASE_DIR, "..", "..", "..",  "..", "Data", "batch_2", "test")

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


    batch2_train = datasets.ImageFolder(batch2_train_dir, transform)
    batch2_valid = datasets.ImageFolder(batch2_valid_dir, transform)
    batch2_test = datasets.ImageFolder(batch2_test_dir, transform)
    batch3_train = datasets.ImageFolder(batch3_train_dir, transform)
    batch3_valid = datasets.ImageFolder(batch3_valid_dir, transform)
    batch3_test = datasets.ImageFolder(batch3_test_dir, transform)
    combined_data = torch.utils.data.ConcatDataset([batch2_train, batch3_train])

    train_loader = DataLoader(dataset=combined_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for images, targets in tqdm(train_loader):
        # scale image to be between 0 and 1
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


if __name__ == '__main__':
    main()

# ============================ Results ============================
############ Batch2+Batch3 Whole ############
# norm_mean = [0.2152, 0.2152, 0.2152]
# norm_std = [0.1313, 0.1313, 0.1313]

############ Batch2 trainSet ############
# norm_mean = [0.2203, 0.2203, 0.2203]
# norm_std = [0.1407, 0.1407, 0.1407]

############ Batch2+Batch3 trainSet ############
# norm_mean = [0.2152, 0.2152, 0.2152]
# norm_std = [0.1313, 0.1313, 0.1313]

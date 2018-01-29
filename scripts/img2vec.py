#!/usr/bin/env python
"""

Calculates the ResNet 50 features
across patches across all images.

"""

import torch, os, glob, numpy as np, pdb, cv2, psycopg2, re
from skimage.feature import hog
from skimage import color
from skimage import io
from skimage.util.shape import view_as_windows
from tqdm import tqdm
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.files = glob.glob(os.path.join(image_directory, '*.jpg'))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cache = {}

    def __len__(self):
        return len(self.files) * 5 * 7

    def __getitem__(self, idx):
        img_id = idx / len(self.files)
        idx = idx % 35
        row = (idx / 7) * 300
        col = (idx % 7) * 300

        if col + 300 > 2000:
            col = 1700

        if img_id in self.cache:
            image = self.cache[img_id]
        else:
            image = cv2.imread(self.files[img_id])
            self.cache[img_id] = image

        subset = image[row:row+300, col:col+300, :]
        res = self.transform(subset)

        return res, (row, col, self.files[img_id])


def collate(batch):
    inputs, meta = zip(*batch)
    return torch.stack(inputs, 0), meta


def main(image_directory, cuda=False):
    """
    Create vector representations for each image.  The resulting numpy matrix
    will contain a row for each patch of an image.  Each image is split up into 
    300x300 patches.  The matrix looks like:

    |row|col|image_id|v1|v2|.......|v_n|
    |---|---|--------|--|--|-------|---|

    Where v1, v2, ..., v_2 are the elements of the vector
    """

    BATCH_SIZE = 256

    if cuda:
        model = torch.nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1]).cuda()
    else:
        model = torch.nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
    model.eval()

    all_vectors = []

    dataset = Dataset(image_directory)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         collate_fn=collate,
                                         num_workers=4)

    for inputs, meta in tqdm(loader):
        if cuda:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        vectors = model(inputs).cpu().data.numpy()
        meta = map(lambda x: (x[0],
                              x[1],
                              int(re.search('image_(\d+).jpg',
                                            x[2]).group(1))),
                   meta)
        print(meta)
        print(vectors)
        all_vectors.append(
            np.concatenate(
                [np.array(meta), vectors.squeeze()],
                axis=1
            )
        )

    all_vectors = np.concatenate(all_vectors)
    np.save('vectors.npy', all_vectors)


if __name__ == "__main__":
    image_dir = '../../../data/images/western'
    cuda = input('Are you using CUDA? (Y/N) ')
    cuda = cuda.upper()
    if cuda == 'Y':
        data = main(image_dir, cuda=True)
    else:
        data = main(image_dir, cuda=False)


# EOF

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from .io import load_tiled_image, preprocess_brightfield, preprocess_segim
from .augmentation import Augmenter


class ImgLblInMem(Sequence):

    def __init__(self, imglbl_paths, batch_size, aug):
        self.batch_size = batch_size

        assert isinstance(aug, Augmenter), '"aug" must be an "Augmenter" object'
        self.aug = aug

        # Load all of the source images into memory
        self.img_lbl_info = []
        for img_path, lbl_path in imglbl_paths:
            img, _ = load_tiled_image(img_path)
            lbl, info = load_tiled_image(lbl_path)
            img = preprocess_brightfield(img)
            lbl = lbl > 0
            self.img_lbl_info.append((img, lbl, info))

        # Initialise ordering
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_lbl_info) / float(self.batch_size)))

    def __getitem__(self, idx):
        Nbatch = self.batch_size
        current_batch = self.ordering[idx*Nbatch : (idx+1)*Nbatch]

        img_batch = []
        lbl_batch = []

        for i in current_batch:
            img, lbl, info = self.img_lbl_info[i]
            img, lbl = self.aug.apply(img, lbl)
            lbl = preprocess_segim(lbl, info)
            img_batch.append(img)
            lbl_batch.append(lbl)

        img_batch = np.array(img_batch)
        lbl_batch = np.array(lbl_batch)
        lbl_batch = np.split(lbl_batch, lbl_batch.shape[-1], axis=-1)

        return img_batch, lbl_batch

    def on_epoch_end(self):
        # Shuffle samples for next epoch
        Nsamples = len(self.img_lbl_info)
        self.ordering = np.random.choice(Nsamples, Nsamples, replace=False)

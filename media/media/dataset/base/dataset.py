import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

class BaseDataset(Dataset):
    """
    A custom dataset class that extends torch.utils.data.Dataset.
    It handles datasets for image classification tasks where each instance consists of:
    - One or more image files,
    - A corresponding set of labels.
    """

    def __init__(self, dataframe, file_columns, label_columns, transform=None):
        """
        Initializes the dataset object.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing the file paths and labels.
            file_columns (list of str or str): Column name(s) in the DataFrame that contain the file paths.
            label_columns (list of str or str): Column name(s) in the DataFrame that contain the labels.
            transform (callable, optional): Optional transform to be applied on a sample (image).
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.file_columns = file_columns if isinstance(file_columns, list) else [file_columns]
        self.label_columns = label_columns if isinstance(label_columns, list) else [label_columns]
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The total number of items in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index idx.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the images and their corresponding labels.
        """
        images = self.load_images(idx)
        labels = self.load_labels(idx)

        if self.transform:
            images = [self.apply_transform(image) for image in images]

        # Return a single image or a list of images based on the input
        images = images[0] if len(images) == 1 else images

        return images, labels

    def load_images(self, idx):
        """
        Loads the images from file paths specified in the DataFrame for a given index.

        Args:
            idx (int): Index of the item to retrieve images for.

        Returns:
            list: A list of images.
        """
        images = []
        for file_column in self.file_columns:
            img_path = self.dataframe.iloc[idx][file_column]
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            images.append(image)
        return images

    def load_labels(self, idx):
        """
        Loads the labels from the DataFrame for a given index.

        Args:
            idx (int): Index of the item to retrieve labels for.

        Returns:
            torch.Tensor: A tensor of labels.
        """
        try:
            label_data = self.dataframe.iloc[idx][self.label_columns]
            labels = label_data.apply(pd.to_numeric, errors="coerce").fillna(0)  # Convert non-numeric values to NaN, then fill NaN with 0
            labels = torch.tensor(labels.values, dtype=torch.float32)
            labels = torch.squeeze(labels)  # Remove extraneous dimensions
        except:
            print(idx, label_data)
        return labels

    def apply_transform(self, image):
        """
        Applies the specified transformations to the image.

        Args:
            image (np.array): An image to apply transformations to.

        Returns:
            np.array: A transformed image.
        """
        augmented = self.transform(image=image)
        image = augmented["image"] if isinstance(self.transform, A.Compose) else augmented
        return image

    def show_samples(self, num_samples=4):
        """
        Displays a sample of num_samples images from the dataset along with their labels.

        Args:
            num_samples (int): Number of samples to display.
        """
        if num_samples <= 0: return None
        figure, ax = plt.subplots(nrows=num_samples, ncols=1, figsize=(5, 5 * num_samples))
        if num_samples ==1: ax = [ax]
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.dataframe))
            images, labels = self[idx]

            image = images if not isinstance(images, list) else images[0]

            if image.shape[0] == 3:  image = image.permute(1, 2, 0)  # Convert CHW to HWC format if necessary

            ax[i].imshow(image)
            ax[i].set_title(f"Labels: {labels}")
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()


from sklearn.preprocessing import LabelEncoder
class DatasetMeta(BaseDataset):
    def __init__(self, dataframe, file_columns, label_columns, transform=None):
        super().__init__(dataframe, file_columns, label_columns, transform)
        self.dataframe = dataframe.reset_index(drop=True)
        self.file_columns = file_columns if isinstance(file_columns, list) else [file_columns]
        self.label_columns = label_columns if isinstance(label_columns, list) else [label_columns]
        self.transform = transform

        self.dataframe['sex'] = self.dataframe['sex'].str.lower().map({'male': 1, 'female': 0}).fillna(-1)
        # self.dataframe['is_norm'] = self.dataframe['Normal']
        self.dataframe['age'].loc[(self.dataframe['age']>90)] = 90
        cols = ['age', 'height', 'weight']
        self.dataframe[cols] = self.dataframe[cols].fillna(self.dataframe[cols].median())
        self.dataframe[cols] = self.dataframe[cols]/100

    def __getitem__(self, idx):
        images = self.load_images(idx)
        metadata = self.load_metadata(idx)
        labels = self.load_labels(idx)

        if self.transform:
            images = [self.apply_transform(image) for image in images]

        # Return a single image or a list of images based on the input
        images = images[0] if len(images) == 1 else images

        return (images, metadata), labels
        # return metadata, labels

    def load_metadata(self, idx):
        metadata = self.dataframe.iloc[idx][['sex', '[18-25)', '[25-30)', '[30-35)',
       '[35-40)', '[40-45)', '[45-50)', '[50-55)', '[55-60)', '[60-65)',
       '[65-70)', '[70-75)', '[75-80)', '[80-85)', '[85-inf))', 'etnia']].values.astype(np.float32)
        return metadata

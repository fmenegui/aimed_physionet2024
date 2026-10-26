import os
from torch.utils.data import DataLoader
import lightning as L

class BaseDataloader(L.LightningDataModule):
    """
    A base data loader module that handles the creation of PyTorch DataLoaders for a given dataset.

    Attributes:
        dataset_class (Dataset): The dataset class to use for creating datasets.
        dataframe (DataFrame): The dataframe containing the data.
        file_columns (str): The name of the column containing file paths.
        label_columns (list of str): List of column names containing the labels.
        batch_size (int): The size of the batch.
        train_indices (list of int): Indices for training data.
        val_indices (list of int): Indices for validation data.
        test_indices (list of int): Indices for test data.
        transforms (dict): Dictionary containing transformations for 'train', 'valid', and 'test'.
        shuffle (bool): Whether to shuffle the data.
        pin_memory (bool): Whether to use pinned memory.
        num_workers (int): The number of workers for data loading.
    """

    def __init__(
        self,
        dataset_class,
        dataframe,
        file_column=None,
        label_columns=None,
        group_column=None,
        batch_size=1,
        num_splits=5,
        train_indices=None,
        val_indices=None,
        test_indices=None,
        transforms=None,
        shuffle=False,
        pin_memory=False,
        num_workers=-1
    ):
        """
        Initializes the BaseDataloader with the provided parameters.
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.dataframe = dataframe
        self.file_columns = file_column
        self.label_columns = label_columns
        self.group_column = group_column
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.transforms = transforms or {}
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers if num_workers >= 0 else (os.cpu_count() or 1)

    def get_dataset(self, indices, stage="train"):
        """
        Creates a dataset using the specified indices and transformation.

        Args:
            indices (list of int): The indices of the data to include in the dataset.
            stage (str): The stage of the dataset ('train', 'valid', or 'test').

        Returns:
            Dataset: The created dataset.
        """
        dataset = self.dataset_class(
            dataframe=self.dataframe.iloc[indices].reset_index(drop=True),
            file_columns=self.file_columns,
            label_columns=self.label_columns,
            transform=self.transforms.get(stage)
        )
        return dataset

    def get_dataloader(self, indices, stage="train"):
        """
        Creates a DataLoader for the given indices and stage.

        Args:
            indices (list of int): The indices of the data to include in the dataloader.
            stage (str): The stage of the dataloader ('train', 'valid', or 'test').

        Returns:
            DataLoader: The created DataLoader.
        """
        if indices is None or len(indices) == 0:
            print(
                "Warning: No indices provided for dataloader. Returning an empty DataLoader."
            )
            return DataLoader([])

        dataset = self.get_dataset(indices, stage)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if stage == "train" else False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """
        Creates a DataLoader for the training data.

        Returns:
            DataLoader: The DataLoader for the training data.
        """
        return self.get_dataloader(self.train_indices, stage="train")

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation data.

        Returns:
            DataLoader: The DataLoader for the validation data.
        """
        return self.get_dataloader(self.val_indices, stage="valid")

    def test_dataloader(self):
        """
        Creates a DataLoader for the test data.

        Returns:
            DataLoader: The DataLoader for the test data.
        """
        return self.get_dataloader(self.test_indices, stage="test")

    def show_batch(self, stage='train'):
        """
        Visualizes a batch of data from the specified stage.

        Args:
            stage (str): The stage of the DataLoader to visualize the batch from ('train', 'valid', or 'test').
        """
        import matplotlib.pyplot as plt
        # Fetch a batch of data
        if stage == 'train':
            dataloader = self.train_dataloader()
        elif stage == 'valid':
            dataloader = self.val_dataloader()
        elif stage == 'test':
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Get a batch from the dataloader
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        images = images[0] if isinstance(images, list) else images

        # Plot the images
        num_images = images.shape[0]
        if num_images <= 0: return None
        figure, ax = plt.subplots(nrows=num_images, ncols=1, figsize=(5, 5 * num_images))
        if num_images ==1: ax = [ax]

        for idx in range(num_images):
            image, label = images[idx], labels[idx]

            if image.shape[0] == 3:  image = image.permute(1, 2, 0)  # Convert CHW to HWC format if necessary

            ax[idx].imshow(image)
            ax[idx].set_title(f"Label: {label}")
            ax[idx].axis("off")
        plt.tight_layout()
        plt.show()

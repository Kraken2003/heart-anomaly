import torch
from torch.utils.data import Dataset
import pandas as pd

class ECG_Data(Dataset):
    """
    Custom PyTorch dataset for loading ECG data.

    This dataset class is designed to handle ECG data with associated demographic information.

    Args:
        df (pandas.DataFrame): DataFrame containing the dataset information.
        window_size (int): Size of the data window for processing ECG data.

    Attributes:
        df (pandas.DataFrame): The input DataFrame containing ECG data and demographic information.
        window_size (int): The specified size of the data window for processing.
    """

    def __init__(self, df, window_size):
        """
        Initializes the ECG_Data dataset.

        Args:
            df (pandas.DataFrame): DataFrame containing the dataset information.
            window_size (int): Size of the data window for processing ECG data.
        """
        self.df = df
        self.window_size = window_size

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing windowed_lead_data (ECG data), label (ECG label),
                   age (demographic information), and sex (demographic information).
        """
        # Extract demographic information
        age = torch.tensor([int(self.df.iloc[index, 1])])
        sex = torch.tensor([0]) if self.df.iloc[index, 2] == 'Male' else torch.tensor([1])
        label = torch.tensor([int(self.df.iloc[index, 3])])

        # Load ECG data from file
        file_name = "/content/outputs_6leads/" + self.df.iloc[index, 0] + ".csv"
        data = pd.read_csv(file_name)
        lead_data = data.iloc[:, 1:7].values.astype(float)

        # Windowing the lead data
        windowed_lead_data = []
        for i in range(0, len(lead_data) - self.window_size + 1, self.window_size):
            windowed_lead_data.append(lead_data[i:i+self.window_size])
        windowed_lead_data = torch.Tensor(windowed_lead_data)
        windowed_lead_data = torch.sum(windowed_lead_data, dim=1)

        return windowed_lead_data, label, age, sex

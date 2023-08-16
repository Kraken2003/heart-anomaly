import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A neural network model for processing ECG data.

    Args:
        input_size (int): Size of the input feature vector.
        hidden_size (int): Size of the hidden layer in the LSTM.
        num_layers (int): Number of LSTM layers.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        hidden_size (int): Size of the hidden layer in the LSTM.
        num_layers (int): Number of LSTM layers.
        device (str): Device for computation ('cpu' or 'cuda').
        lstm (nn.LSTM): Bidirectional LSTM layer.
        fc (nn.Sequential): Fully connected layers for classification.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        """
        Initializes the Model.

        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Size of the hidden layer in the LSTM.
            num_layers (int): Number of LSTM layers.
            device (str): Device for computation ('cpu' or 'cuda').
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 500),
            nn.Linear(500, 2)
        )
        self.sigmoid = nn.Sigmoid()

    def positional_encoding(self, var, size):
        """
        Generates positional encoding for input sequences.

        Args:
            var (torch.Tensor): Input tensor.
            size (int): Size of the positional encoding.

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, size, 2, device=self.device).float() / size))
        pos_enc_a = torch.sin(var.repeat(1, size // 2) * inv_freq)
        pos_enc_b = torch.cos(var.repeat(1, size // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, lead_data, age, sex):
        """
        Forward pass of the model.

        Args:
            lead_data (torch.Tensor): Input ECG data.
            age (torch.Tensor): Age information.
            sex (torch.Tensor): Gender information.

        Returns:
            torch.Tensor: Model output.
        """
        batch_size = lead_data.size(0)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)

        lstm_out, _ = self.lstm(lead_data, (h0, c0))
        last_output = lstm_out[:, -1, :]

        age = self.positional_encoding(age, 64)
        sex = self.positional_encoding(sex, 64)

        combined = torch.cat((last_output, age, sex), dim=1)

        output = self.fc(combined)
        output = self.sigmoid(output)

        return output

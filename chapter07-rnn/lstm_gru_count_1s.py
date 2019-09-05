import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gru_cell import GRUCell
from lstm_cell import LSTMCell

EPOCHS = 10  # training epochs
TRAINING_SAMPLES = 10000  # training dataset size
BATCH_SIZE = 16  # mini batch size
TEST_SAMPLES = 1000  # test dataset size
SEQUENCE_LENGTH = 20  # binary sequence length
HIDDEN_UNITS = 20  # hidden units of the LSTM cell


class LSTMModel(nn.Module):
    """LSTM model with a single output layer connected to the lstm cell output"""

    def __init__(self, input_dim, hidden_size, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = LSTMCell(input_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Start with empty network output and cell state to initialize the sequence
        c_t = torch.zeros((self.hidden_size, x.size(0), self.hidden_size)).to(x.device)
        h_t = torch.zeros((self.hidden_size, x.size(0), self.hidden_size)).to(x.device)

        c_t, h_t = c_t[0, :, :], h_t[0, :, :]

        for seq in range(x.size(1)):
            h_t, c_t = self.lstm(x[:, seq, :], (h_t, c_t))

        # Remove unnecessary dimensions
        out = h_t.squeeze()

        # Final output layer
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    """LSTM model with a single output layer connected to the lstm cell output"""

    def __init__(self, input_dim, hidden_size, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size

        self.gru = GRUCell(input_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Start with empty network output and cell state to initialize the sequence
        h_t = torch.zeros((self.hidden_size, x.size(0), self.hidden_size)).to(x.device)

        h_t = h_t[0, :, :]

        for seq in range(x.size(1)):
            h_t = self.gru(x[:, seq, :], h_t)

        # Remove unnecessary dimensions
        out = h_t.squeeze()

        # Final output layer
        out = self.fc(out)

        return out


def generate_dataset(sequence_length: int, samples: int):
    """
    Generate training/testing datasets
    :param sequence_length: length of the binary sequence
    :param samples: number of samples
    """

    sequences = list()
    labels = list()
    for i in range(samples):
        a = np.random.randint(sequence_length) / sequence_length
        sequence = list(np.random.choice(2, sequence_length, p=[a, 1 - a]))
        sequences.append(sequence)
        labels.append(int(np.sum(sequence)))

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int8)

    result = torch.utils.data.TensorDataset(
        torch.from_numpy(sequences).float().unsqueeze(-1),
        torch.from_numpy(labels).float())

    return result


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        model.zero_grad()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(outputs.round() == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(outputs.round() == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="LSTM/GRU count 1s in binary sequence")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-lstm', action='store_true', help="LSTM")
    group.add_argument('-gru', action='store_true', help="GRU")
    args = parser.parse_args()

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate training and testing datasets
    train = generate_dataset(SEQUENCE_LENGTH, TRAINING_SAMPLES)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    test = generate_dataset(SEQUENCE_LENGTH, TEST_SAMPLES)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate LSTM or GRU model
    # input of size 1 for digit of the sequence
    # number of hidden units
    # regression model output size (number of ones)
    if args.lstm:
        model = LSTMModel(1, HIDDEN_UNITS, 1)
    elif args.gru:
        model = GRUModel(1, HIDDEN_UNITS, 1)

    # Transfer the model to the GPU
    model = model.to(device)

    # loss function (we use MSELoss because of the regression)
    loss_function = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Train
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, test_loader)

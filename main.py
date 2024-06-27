import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from model import LSTMClassifier


def train_model():
    input_size = padded_tensor_train.shape[2]  # number of data features, should be 12
    hidden_size = 64  # HP, we'll adjust as we train
    # num_layers = 1  # also HP we can add to the architecture after we train initially
    output_size = 9  # number of classes that the lstm is trying to predict

    model = LSTMClassifier(input_size, hidden_size, output_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(padded_tensor_train)
        loss = loss_function(outputs, tensor_train_labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# The data is formatted very stupidly.
# ae.test is a series of 12 input values followed by 12 output values.
# What seperates them? A series of 1.0 to indicate the end of the recording. And a new line.

# ae.train is the same!!! EXCEPT WITHOUT NEW LINES FOR SOME REASON


# Read these files
def read_txt_file(filename):
    inputs = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        current_input = []
        for line in lines:
            values = line.strip().split()
            # If not the end of a record
            if values and values[0] != '1.0':
                # here I removed the part where we are doing the first 12 inputs
                # because I am not sure it was doing anything
                input_values = [float(val) if val else np.nan for val in values]
                current_input.append(input_values)
            # We're at the end
            elif values and values[0] == '1.0':
                inputs.append(current_input)
                current_input = []
    return inputs


def create_training_labels():
    training_labels = []
    for x in range(9):
        for y in range(30):
            training_labels.append(x)

    return training_labels


# Read the files
train_inputs = read_txt_file('ae.train')
test_inputs = read_txt_file('ae.test')
train_labels = create_training_labels()

tensor_train = [torch.tensor(seq, dtype=torch.float32) for seq in train_inputs]
tensor_test = [torch.tensor(seq, dtype=torch.float32) for seq in test_inputs]
tensor_train_labels = torch.tensor(create_training_labels())


padded_tensor_train = pad_sequence(tensor_train)
padded_tensor_test = pad_sequence(tensor_test)

print(padded_tensor_train.size())
train_model()


if __name__ == "__main__":
    pass

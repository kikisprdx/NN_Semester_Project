import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMClassifier


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
        # Line by line we strip and split all values
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

# this conversion doesn't work bcs of the variable size of the input
# so far the biggest issue in terms of using variable size input
# which is not padded
tensor_train = torch.tensor(train_inputs)
tensor_test = torch.tensor(test_inputs)
tensor_train_labels = torch.tensor(train_labels)

input_size = tensor_train.shape[2]  # number of data features, should be 12
hidden_size = 64  # HP, we'll adjust as we train
# num_layers = 1  # also HP we can add to the architecture after we train initially
output_size = 8  # number of classes that the lstm is trying to predict

model = LSTMClassifier(input_size, hidden_size, output_size)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(tensor_train)
    loss = loss_function(outputs, tensor_train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# READ: Different recording have different lengths
# Do we a) shorten the recordings to the shortest one,
# b) pad the recordings to the longest one, or
# c) something else?
# For now I'm doing b) but keep that in mind


if __name__ == "__main__":
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from model import LSTMClassifier


def plot_loss(num_epochs, train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_model():
    input_size = padded_tensor_train.shape[2]  # number of data features, should be 12
    hidden_size = 64  # HP, we'll adjust as we train
    # num_layers = 1  # also HP we can add to the architecture after we train initially
    output_size = 9  # number of classes that the lstm is trying to predict

    model = LSTMClassifier(input_size, hidden_size, output_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(padded_tensor_train)
        loss = loss_function(outputs, tensor_train_labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    plot_loss(num_epochs, train_losses)

    return model


def test_model(model):
    model.eval()
    with torch.no_grad():
        test_outputs = model(padded_tensor_test)
        test_loss = nn.CrossEntropyLoss()(test_outputs, tensor_test_labels)
        test_predictions = torch.argmax(test_outputs, dim=1)
        accuracy = accuracy_score(tensor_test_labels.numpy(), test_predictions.numpy())
        f1 = f1_score(tensor_test_labels.numpy(), test_predictions.numpy(), average='weighted')

    print(f'Test Loss: {test_loss.item()}')
    print(f'Test Accuracy: {accuracy}')
    print(f'Test F1 Score: {f1}')
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


# isn't pretty but it works
def create_testing_labels():
    testing_labels = []
    for signal_1 in range(31):
        testing_labels.append(0)
    for signal_2 in range(35):
        testing_labels.append(1)
    for signal_3 in range(88):
        testing_labels.append(2)
    for signal_4 in range(44):
        testing_labels.append(3)
    for signal_5 in range(29):
        testing_labels.append(4)
    for signal_6 in range(24):
        testing_labels.append(5)
    for signal_7 in range(40):
        testing_labels.append(6)
    for signal_8 in range(50):
        testing_labels.append(7)
    for signal_9 in range(29):
        testing_labels.append(8)

    return testing_labels


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
test_labels = create_testing_labels()

tensor_train = [torch.tensor(seq, dtype=torch.float32) for seq in train_inputs]
tensor_test = [torch.tensor(seq, dtype=torch.float32) for seq in test_inputs]
tensor_train_labels = torch.tensor(train_labels, dtype=torch.long)
tensor_test_labels = torch.tensor(test_labels, dtype=torch.long)

padded_tensor_train = pad_sequence(tensor_train)
padded_tensor_test = pad_sequence(tensor_test)

model = train_model()
test_model(model)

if __name__ == "__main__":
    pass

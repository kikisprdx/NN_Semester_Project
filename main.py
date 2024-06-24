import numpy as np

# The data is formatted very stupidly. 
# ae.test is a series of 12 input values followed by 12 output values.
# What seperates them? A series of 1.0 to indicate the end of the recording. And a new line. 

# ae.train is the same!!! EXCEPT WITHOUT NEW LINES FOR SOME REASON 


# Read these files
def read_txt_file(filename):

    inputs = []
    outputs = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        current_input = []
        current_output = []
        # Line by line we strip and split all values 
        for line in lines:
            values = line.strip().split()
            # If not the end of a record 
            if values and values[0] != '1.0':
                # We add the whole line (if invalid its going to be set to NaN) 
                # First 12 set of values are input, the next 12 are output
                # I only got this by looking through the original matlab file and inferencing this fact
                # I could be wrong so maybe I'll ask the professor 
                input_values = [float(val) if val else np.nan for val in values[:12]]
                output_values = [float(val) if val else np.nan for val in values[12:]]
                current_input.append(input_values)
                current_output.append(output_values)
            # We're at the end
            elif values and values[0] == '1.0':
                inputs.append(current_input)
                outputs.append(current_output)
                current_input = []
                current_output = []
    return inputs, outputs


# Read the files
train_inputs, train_outputs = read_txt_file('ae.train')
test_inputs, test_outputs = read_txt_file('ae.test')

train_outputs = []
for i in range(269):
    speaker_index = (i // 30) + 1  # Assuming 9 speakers, each with 30 time series
    print(train_inputs[i])
    l = len(train_inputs[i])
    teacher = np.zeros((l, 9))
    teacher[:, speaker_index - 1] = 1  # One-hot encoding for speaker index
    train_outputs.append(teacher)

# Create teacher signals for test data
test_outputs = []
speaker_index = 1
block_counter = 0
block_lengths = [31, 35, 88, 44, 29, 24, 40, 50, 29]  # Assuming the same block lengths as in MATLAB code
for i in range(370):
    block_counter += 1
    if block_counter > block_lengths[speaker_index - 1]:
        speaker_index += 1
        block_counter = 1
    l = len(test_inputs[i])
    teacher = np.zeros((l, 9))
    teacher[:, speaker_index - 1] = 1  # One-hot encoding for speaker index
    test_outputs.append(teacher)

# READ: Different recording have different lengths 
# Do we a) shorten the recordings to the shortest one, 
# b) pad the recordings to the longest one, or
# c) something else?
# For now I'm doing b) but keep that in mind
max_len_train_inputs = max(len(ts) for ts in train_inputs)
max_len_train_outputs = max(len(ts) for ts in train_outputs)
max_len_test_inputs = max(len(ts) for ts in test_inputs)
max_len_test_outputs = max(len(ts) for ts in test_outputs)

train_inputs = [np.pad(ts, ((0, max_len_train_inputs - len(ts)), (0, 0)), mode='constant', constant_values=np.nan) for ts in train_inputs]
train_outputs = [np.pad(ts, ((0, max_len_train_outputs - len(ts)), (0, 0)), mode='constant', constant_values=np.nan) for ts in train_outputs]
test_inputs = [np.pad(ts, ((0, max_len_test_inputs - len(ts)), (0, 0)), mode='constant', constant_values=np.nan) for ts in test_inputs]
test_outputs = [np.pad(ts, ((0, max_len_test_outputs - len(ts)), (0, 0)), mode='constant', constant_values=np.nan) for ts in test_outputs]

train_inputs = np.array(train_inputs)
test_inputs = np.array(test_inputs)
train_outputs = np.array(train_outputs)
test_outputs = np.array(test_outputs)

# BOOM 
print(train_inputs.shape)
print(test_inputs.shape)
print(train_outputs.shape)
print(test_outputs.shape)

#print(train_inputs)
# print(train_outputs)
print(test_outputs)
if __name__ == "__main__":
    pass

import matlab.engine 
import os

eng = matlab.engine.start_matlab()
cwd = os.getcwd()
eng.cd(cwd, nargout=0)
eng.aeDataImport(nargout=0)
train_inputs = eng.workspace['trainInputs']
test_inputs = eng.workspace['testInputs']
train_outputs = eng.workspace['trainOutputs']
test_outputs = eng.workspace['testOutputs']

if __name__ == "__main__":
    pass

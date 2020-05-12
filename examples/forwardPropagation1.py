import torch

'''
Either use numbers from example
or you create your own numbers
'''

'''
Numbers from example:
'''


def sigmoid(x):
    return 1/(1+torch.exp(-1*x))

def h1_1(S, weights, bias, activation):
    assert len(S) == len(weights), "These should be the same length"

    signal = torch.zeros(1)

    for i in range(len(S)):
        signal += S[i]*weights[i]

    signal += bias


    return activation(signal)

def O_1(S, weights):
    assert len(S) == len(weights), "These should be the same length"

    signal = 0

    for i in range(len(S)):
        signal += S[i]*weights[i]

    return signal

i_1 = 5
i_2 = 4
i_3 = 2

def run_example():
    hiddenSignal = h1_1(torch.tensor([i_1, i_2, i_3 ]), torch.tensor([0.4595,  0.1161,  0.3385]), torch.tensor([-0.0510]), sigmoid)

    print("signal: {}".format(hiddenSignalsignal))

    outputSignal = O_1(hiddenSignal, torch.tensor([0.0423]))

    print("[y-hat value]: {}".format(outputSignal))


run_example()

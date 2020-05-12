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

def identity(x):
    return x

def h1_n(S, weights, bias, activation):
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
    '''
    $i_1 = 5$, $i_2 = 4$, $i_3 = 2$
    % 0.4595,  0.1161,  0.3385
    $w1^1_1 = 0.4595$, $w1^1_2 = 0.1161$, $w1^1_3 = 0.3385$

    % 0.2495,  -0.1181,  0.2340
    $w1^2_1 = 0.2495$, $w1^2_2 = -0.1181$, $w1^2_3 = 0.2340$

    % -0.0510, -0.1510
    $b1_1 = -0.0510$, $b1_2 = -0.1510$

    $w2^1_1 = 0.4595$, $w2^1_2 = 0.7261$,

    % 0.2495,  -0.1181,  0.2340
    $w2^2_1 = 0.2785$, $w2^2_2 = 0.9729$,

    % -0.0510, -0.1510
    $b2_1 = -0.0510$, $b2_2 = -0.1510$

    % 0.0423, -0.2101
    $W1^1_1 = 0.0423$, $W1^1_2 = -0.2101$
    '''



    h1_1 = h1_n(torch.tensor([i_1, i_2, i_3 ]), torch.tensor([0.4595,  0.1161,  0.3385]), torch.tensor([-0.0510]), sigmoid)
    h1_2 = h1_n(torch.tensor([i_1, i_2, i_3 ]), torch.tensor([0.2495,  -0.1181,  0.2340]), torch.tensor([-0.1510]), sigmoid)


    h2_1 = h1_n(torch.tensor([h1_1, h1_2]), torch.tensor([0.4595,  0.7261]), torch.tensor([-0.0510]), sigmoid)
    h2_2 = h1_n(torch.tensor([h1_1, h1_2]), torch.tensor([0.2785,  0.9729]), torch.tensor([-0.1510]), sigmoid)
    print(h2_1)
    print(h2_2)

    O_1 = h1_n(torch.tensor([h1_1, h1_2]), torch.tensor([0.0423,  -0.2101]), torch.tensor([0.]), identity)

    print(O_1)

    # h1_2 = h1_n(torch.tensor([i_1, i_2, i_3 ]), torch.tensor([0.2495,  -0.1181,  0.2340]), torch.tensor([-0.1510]), sigmoid)
    # print(h1_2)
    #
    # outputSignal = O_1([h1_1, h1_2], torch.tensor([0.0423, -0.2101]))
    #
    # print("[y-hat value]: {}".format(outputSignal))


run_example()

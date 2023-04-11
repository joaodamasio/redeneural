import numpy as np

input_training = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1],
                           [1, 0, 0]])

output_training = np.array([[0, 1, 1, 0, 1]]).T

print(output_training)

np.random.seed(1)

sinapses_weights = 2 * np.random.random((3,1)) -1



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoide_derivate(x):
    return x*(1-x)

for i in range (10000):
    input_layer = input_training
    output = sigmoid(np.dot(input_layer, sinapses_weights))

    error = output_training - output
    error_ajust = error * sigmoide_derivate(output)

    sinapses_weights += np.dot(input_layer.T,error_ajust)

print(output)
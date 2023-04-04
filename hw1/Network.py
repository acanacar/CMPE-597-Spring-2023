import numpy as np


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = np.random.normal(0,1/n_neurons,size=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def forward_embedding(self, input1, input2, input3):
        '''

        :param input1: first word with shape of (batch size x 250)
        :param input2: second word with shape of (batch size x 250)
        :param input3: third word with shape of (batch size x 250)
        :return:
        '''
        embedding_1 = np.dot(input1, self.weights)  # embedding of first word (batch size x 16)
        embedding_2 = np.dot(input2, self.weights)  # embedding of second word (batch size x 16)
        embedding_3 = np.dot(input3, self.weights)  # embedding of third word (batch size x 16)
        # self.output = np.hstack((embedding_1, embedding_2, embedding_3))
        self.output = np.concatenate((embedding_1, embedding_2, embedding_3),axis=1)

    def backward_embedding(self, input1, input2, input3, dvalues):
        w1_gradient = np.zeros((250, 16))  # initialization for gradients of weight 1 (250x16)
        # Inverse of concatanation while calculating w1_gradients: Split into three
        w1_gradient += np.dot(input1.T, dvalues[:, 0:16])
        w1_gradient += np.dot(input2.T, dvalues[:, 16:32])
        w1_gradient += np.dot(input3.T, dvalues[:, 32:48])
        self.dweights = w1_gradient

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# Sigmoid activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from inputs
        self.output = 1 / 1 + np.exp(-inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot encoded convert them
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Gradient calculation
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples  # Normalize our gradient
        # print('self.dinputs of softmax' ,self.dinputs)


class CategoricalCrossentropyLoss:
    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        # y_pred_clipped =y_pred.copy()


        if len(y_true.shape) == 1: #  if categorical labels
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: #one-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        avg_sample_loss = np.mean(sample_losses)
        return avg_sample_loss


class Network:
    def __init__(self):
        self.embedding_layer = None
        self.hidden_layer = None
        self.loss = None

    def create_loss_function(self):
        self.loss = CategoricalCrossentropyLoss()

    def create_embedding_layer(self, n_inputs, n_neurons):
        self.embedding_layer = Layer_Dense(n_inputs, n_neurons)  # w1

    def create_hidden_layer(self, n_inputs, n_neurons):
        self.hidden_layer = Layer_Dense(n_inputs, n_neurons)  # w2

    def create_output_layer(self, n_inputs, n_neurons):
        self.output_layer = Layer_Dense(n_inputs, n_neurons)  # w3

    def create_activation_function_for_hidden_layer(self, activation_type):
        if activation_type == 'softmax':
            self.activation_hidden_layer = Activation_Softmax()
        elif activation_type == 'sigmoid':
            self.activation_hidden_layer = Activation_Sigmoid()

    def create_activation_function_for_output_layer(self, activation_type):
        if activation_type == 'softmax':
            self.activation_output_layer = Activation_Softmax()
        elif activation_type == 'sigmoid':
            self.activation_output_layer = Activation_Sigmoid()

    def forward_propagate(self, input1, input2, input3):
        self.embedding_layer.forward_embedding(input1, input2, input3)
        self.hidden_layer.forward(self.embedding_layer.output)  # hidden weight -> W2 (128 x 48)  hidden biases -> b1 (128 x 1 )
        self.activation_hidden_layer.forward(self.hidden_layer.output)  # sigmoid activation function
        self.output_layer.forward(self.activation_hidden_layer.output)  # output layer weights -> W3 ( 250 x 128 ) output layer biases -> b2 (250 x 1)
        self.activation_output_layer.forward(self.output_layer.output)  # softmax activation function

    def backward_propagate(self, input1, input2, input3, batch_targets):
        self.activation_output_layer.backward(dvalues=self.activation_output_layer.output, y_true=batch_targets)  # gradient of loss wrt softmax layer
        self.output_layer.backward(self.activation_output_layer.dinputs)  # calculated w3 x2 b2 gradient
        self.activation_hidden_layer.backward(self.output_layer.dinputs)
        self.hidden_layer.backward(self.activation_hidden_layer.dinputs)  # calculated w2 x1 b1 gradient
        self.embedding_layer.backward_embedding(input1, input2, input3, self.hidden_layer.dinputs)  # calculated w1 gradient

    def update_params(self, learning_rate):
        self.embedding_layer.weights -= learning_rate * self.embedding_layer.dweights
        # print(f"self.embedding_layer.weights : {self.embedding_layer.weights}")
        # print(f"self.hidden_layer.dweights : {self.hidden_layer.dweights}")
        self.hidden_layer.weights -= learning_rate * self.hidden_layer.dweights
        # print(f"self.output_layer.dweights : {self.output_layer.dweights}")
        self.output_layer.weights -= learning_rate * self.output_layer.dweights
        # print(f"self.hidden_layer.dbiases : {self.hidden_layer.dbiases}")
        self.hidden_layer.biases -= learning_rate * self.hidden_layer.dbiases
        # print(f"self.output_layer.dbiases : {self.output_layer.dbiases}")
        self.output_layer.biases -= learning_rate * self.output_layer.dbiases

    def eval_valid_data(self, inputs, targets, no_of_batch, batch_size_validation):
        tot_loss, tot_accuracy = 0., 0.

        for batch_no in range(no_of_batch):
            batch_inputs = inputs[batch_no * batch_size_validation:(batch_no + 1) * batch_size_validation, :]
            input1, input2, input3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
            batch_targets = targets[batch_no * batch_size_validation:(batch_no + 1) * batch_size_validation, :]
            self.forward_propagate(input1, input2, input3)
            tot_loss += self.loss.calculate(self.activation_output_layer.output, batch_targets)
            predictions = np.argmax(self.activation_output_layer.output, axis=1)
            if len(batch_targets.shape) == 2:
                batch_targets = np.argmax(batch_targets, axis=1)
            tot_accuracy += np.mean(predictions == batch_targets)
        avg_loss = tot_loss / no_of_batch
        avg_acc = tot_accuracy / no_of_batch
        return avg_loss, avg_acc

    def eval_test_data(self, inputs, targets):

        input1, input2, input3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        self.forward_propagate(input1, input2, input3)
        avg_loss = self.loss.calculate(self.activation_output_layer.output, targets)

        predictions = np.argmax(self.activation_output_layer.output, axis=1)

        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        avg_acc = np.mean(predictions == targets)
        return avg_loss, avg_acc

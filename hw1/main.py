import pickle

import numpy as np
import argparse

from Network import Network
from pathlib import Path

PROJECT_PATH = Path(fr"C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1")
DATA_PATH = PROJECT_PATH/ Path(fr"hw1/data")


def one_hot_encode(idx, vocab_size):
    res = [0] * vocab_size
    res[idx] = 1
    return np.array(res)


def get_one_hot_encode_matrix(data, train_f):
    if train_f:
        # means that training data so shape will be (length,3,250)
        one_hot_data_matrix = np.zeros((data.shape[0], 3, COUNT_OF_UNIQUE_WORDS))
        for i in range(data.shape[0]):
            for j in range(3):
                one_hot_data_matrix[i, j, data[i, j]] = 1
    else:
        # means that test data so shape will be (length,250
        one_hot_data_matrix = np.zeros((data.shape[0], COUNT_OF_UNIQUE_WORDS))
        for i in range(data.shape[0]):
            one_hot_data_matrix[i, data[i]] = 1

    return one_hot_data_matrix


def create_idx_mapping_for_vocab(vocab_dict):
    # create integer id
    word_to_id = {}
    id_to_word = {}
    for idx, word in enumerate(vocab_dict):
        word_to_id[word] = idx
        id_to_word[idx] = word
    return word_to_id, id_to_word


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",nargs='?',default=40,type=int)
    parser.add_argument("--batch_size_train",nargs='?',default=50,type=int)
    parser.add_argument("--batch_size_valid",nargs='?',default=50,type=int)
    parser.add_argument("--learning_rate",nargs='?',default=0.01,type=float)
    parser.add_argument("--project_path", type=str, help='path of project folder')


    args = parser.parse_args()
    print(args,'\n')
    PROJECT_PATH = Path(args.project_path)
    DATA_PATH = PROJECT_PATH / Path(fr"hw1/data")
    BATCH_SIZE = args.batch_size_train
    BATCH_SIZE_VALIDATION = args.batch_size_valid
    NO_OF_EPOCH = args.epoch
    learning_rate = args.learning_rate
    print(f"project path is {str(PROJECT_PATH)}")
    print(f"{NO_OF_EPOCH} epoch will be executed")
    print(f"batch size for training  {BATCH_SIZE}")
    print(f"batch size for validation  {BATCH_SIZE_VALIDATION}")

    # get data from memory
    vocab_data = np.load(file=str(DATA_PATH / Path(fr'vocab.npy')))
    train_inputs = np.load(file=str(DATA_PATH / Path(fr'train_inputs.npy')))
    train_targets = np.load(file=str(DATA_PATH / Path(fr'train_targets.npy')))
    train_targets = train_targets.reshape(len(train_inputs), 1)
    validation_inputs = np.load(file=str(DATA_PATH / Path(fr'valid_inputs.npy')))
    validation_targets = np.load(file=str(DATA_PATH / Path(fr'valid_targets.npy')))
    no_of_batch = int(len(train_inputs) / BATCH_SIZE)
    no_of_batch_validation = int(len(validation_inputs) / BATCH_SIZE_VALIDATION)
    COUNT_OF_UNIQUE_WORDS = len(vocab_data)  # 250

    # create mapping for 250 words vocab
    word_to_id, id_to_word = create_idx_mapping_for_vocab(vocab_dict=vocab_data)

    # shuffle training data
    training_data = np.concatenate((train_inputs, train_targets), axis=1)
    np.random.seed(0)

    np.random.shuffle(training_data)
    train_inputs, train_targets = training_data[:, 0:3], training_data[:, 3]

    # convert train and validation datas to one hot encoding vectors and put it in matrix
    one_hot_train_inputs = get_one_hot_encode_matrix(data=train_inputs, train_f=1)
    one_hot_train_targets = get_one_hot_encode_matrix(data=train_targets, train_f=0)
    one_hot_validation_inputs = get_one_hot_encode_matrix(data=validation_inputs, train_f=1)
    one_hot_validation_targets = get_one_hot_encode_matrix(data=validation_targets, train_f=0)

    # create network object and set its layers and weights
    network = Network()
    network.create_embedding_layer(250, 16)
    network.create_hidden_layer(48, 128)
    network.create_activation_function_for_hidden_layer(activation_type='sigmoid')
    network.create_output_layer(128, 250)
    network.create_activation_function_for_output_layer(activation_type='softmax')
    network.create_loss_function()
    print(f"trainin is starting with batch_size: {BATCH_SIZE} number of epoch: {NO_OF_EPOCH} number of batch for each epoch: {no_of_batch}")

    history_dict = dict(train_loss_history=[],
                        train_accuracy_history=[],
                        validation_loss_history=[],
                        validation_accuracy_history=[])

    for epoch in range(NO_OF_EPOCH):

        print(f"Epoch({epoch})")

        for batch_no in range(no_of_batch):
            # get inputs and targets for this batch
            batch_inputs = one_hot_train_inputs[batch_no * BATCH_SIZE:(batch_no + 1) * BATCH_SIZE, :]
            batch_targets = one_hot_train_targets[batch_no * BATCH_SIZE:(batch_no + 1) * BATCH_SIZE, :]

            batch_input1, batch_input2, batch_input3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]

            # forward propagation for inputs
            network.forward_propagate(batch_input1, batch_input2, batch_input3)
            # backward and update parameters
            network.backward_propagate(batch_input1, batch_input2, batch_input3, batch_targets)
            network.update_params(learning_rate=learning_rate)

        # calculate loss, predictions and accuracy
        input1 = one_hot_train_inputs[:, 0]
        input2 = one_hot_train_inputs[:, 1]
        input3 = one_hot_train_inputs[:, 2]
        network.forward_propagate(input1, input2, input3)

        train_loss = network.loss.calculate(network.activation_output_layer.output, one_hot_train_targets)

        predictions = np.argmax(network.activation_output_layer.output, axis=1)
        if len(one_hot_train_targets.shape) == 2:  # convert one hot train targets to normal
            y_target_idx = np.argmax(one_hot_train_targets, axis=1)

        train_accuracy = np.mean(predictions == y_target_idx)
        history_dict['train_loss_history'].append(train_loss)
        history_dict['train_accuracy_history'].append(train_accuracy)

        # Check validation accuracy
        loss_valid_data, accuracy_valid_data = network.eval_valid_data(one_hot_validation_inputs, one_hot_validation_targets, no_of_batch_validation, BATCH_SIZE_VALIDATION)
        history_dict['validation_loss_history'].append(loss_valid_data)
        history_dict['validation_accuracy_history'].append(accuracy_valid_data)
        print(f"epoch: {epoch} batch_no:{batch_no} train_acc: {train_accuracy:.4f} train_loss: {train_loss:.4f} validation_acc: {accuracy_valid_data:.4f} validation_loss: {loss_valid_data:.4f}")

    model_params = {}
    model_params['W_embeddings'] = network.embedding_layer.weights
    model_params['W_2'] = network.hidden_layer.weights
    model_params['b1'] = network.hidden_layer.biases
    model_params['W_3'] = network.output_layer.weights
    model_params['b_2'] = network.output_layer.biases
    model_params['history_dict'] = history_dict
    network.history_dict = history_dict
    network.params = {}
    network.params['W_embeddings'] = network.embedding_layer.weights
    network.params['W_2'] = network.hidden_layer.weights
    network.params['b1'] = network.hidden_layer.biases
    network.params['W_3'] = network.output_layer.weights
    network.params['b_2'] = network.output_layer.biases
    pickle.dump(model_params, open(PROJECT_PATH/Path('hw1')/Path('saved_model_params.pkl'), 'wb'))
    pickle.dump(network, open(PROJECT_PATH/Path('hw1')/Path('saved_network_model.pkl'), 'wb'))

    print("saved_model.pkl is saved.")

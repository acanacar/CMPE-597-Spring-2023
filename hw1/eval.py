from pathlib import Path

import numpy as np
import pickle

from cmpe_587_assignment1.main import create_idx_mapping_for_vocab, one_hot_encode

network = pickle.load(open(fr'C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1\saved_network_model.pkl', 'rb'))

data_path = Path(fr"C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1\data")

# Load data

test_inputs = np.load(file=str(data_path / Path(fr'test_inputs.npy')))
test_targets = np.load(file=str(data_path / Path(fr'test_targets.npy')))
test_targets = test_targets.reshape(len(test_inputs), 1)

test_inputs_one_hot = np.zeros((test_inputs.shape[0], 3, 250))
for i in range(test_inputs.shape[0]):
    for j in range(3):
        test_inputs_one_hot[i, j, test_inputs[i, j]] = 1

test_targets_one_hot = np.zeros((test_targets.shape[0], 250))
for i in range(test_targets.shape[0]):
    test_targets_one_hot[i, test_targets[i]] = 1

loss, accuracy = network.eval_test_data(inputs=test_inputs_one_hot, targets=test_targets_one_hot)

print(f"Test data accuracy: {accuracy:.3f}")


def predict_next_item(network, input1, input2, input3):
    vocab_data = np.load(file=str(data_path / Path(fr'vocab.npy')))
    word_to_id, id_to_word = create_idx_mapping_for_vocab(vocab_dict=vocab_data)
    if input1 not in word_to_id or input2 not in word_to_id or input3 not in word_to_id:
        return IndexError('please try another word in vocabulary')
    one_hot_input1 = one_hot_encode(idx=word_to_id[input1], vocab_size=len(vocab_data))
    one_hot_input2 = one_hot_encode(idx=word_to_id[input2], vocab_size=len(vocab_data))
    one_hot_input3 = one_hot_encode(idx=word_to_id[input3], vocab_size=len(vocab_data))

    network.forward_propagate([one_hot_input1], [one_hot_input2], [one_hot_input3])
    predictions = np.argmax(network.activation_output_layer.output, axis=1)
    return vocab_data[predictions[0]]


# City of new
print(f"city of new ->({predict_next_item(network, 'city', 'of', 'new')})<-")
# Life in the
print(f"life in the ->({predict_next_item(network, 'life', 'in', 'the')})<-")
# He is the
print(f"he is the ->({predict_next_item(network, 'he', 'is', 'the')})<-")

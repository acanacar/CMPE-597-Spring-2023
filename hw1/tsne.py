import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pathlib import Path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", type=str, help='path of project folder')
    args = parser.parse_args()
    PROJECT_PATH = Path(args.project_path)
    DATA_PATH = PROJECT_PATH / Path(fr"hw1/data")

    my_model = pickle.load(open(PROJECT_PATH / Path('hw1')/Path(fr'saved_model_params.pkl'), 'rb'))

    vocab_data = np.load(file=str(DATA_PATH / Path(fr'vocab.npy')))

    one_hot_matrix = np.identity(250)

    # my_model = pickle.load(open(fr'C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1\saved_model.pkl','rb'))
    # print("-> saved_model.pkl is loaded.")

    tsne_obj = TSNE(n_components=2)
    embedding_layer_weights = np.dot(one_hot_matrix, my_model['W_embeddings']) # (250x16)
    embedding_layer_weights_2d = tsne_obj.fit_transform(embedding_layer_weights) # (250x2)
    print("-> 2d embedding weights are generated")

    plt.scatter(embedding_layer_weights_2d[:, 0], embedding_layer_weights_2d[:, 1], s=0)

    for label, x, y in zip(vocab_data, embedding_layer_weights_2d[:, 0], embedding_layer_weights_2d[:, 1]):
        plt.annotate(label, xy=(x, y),textcoords='offset points', xytext=(0, 0),size=8 )

    plt.savefig('tsne_result.png')
    plt.show()

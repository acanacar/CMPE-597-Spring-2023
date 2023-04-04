from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pathlib import Path

DATA_PATH = Path(fr"C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1\data")

vocab_data = np.load(file=str(DATA_PATH / Path(fr'vocab.npy')))

one_hot_matrix = np.identity(250)

my_model = pickle.load(open(fr'C:\Users\a.acar\PycharmProjects\NN_from_stratch\cmpe_587_assignment1\saved_model.pkl','rb'))
print("-> model.pk is loaded.")

tsne_obj = TSNE(n_components=2)
embedding_layer_weights = np.dot(one_hot_matrix, my_model['W_embeddings']) # (250x16)
embedding_layer_weights_2d = tsne_obj.fit_transform(embedding_layer_weights) # (250x2)
print("-> 2d embedding weights are generated")
# np.set_printoptions(suppress=True)


plt.scatter(embedding_layer_weights_2d[:, 0], embedding_layer_weights_2d[:, 1], s=0)

for label, x, y in zip(vocab_data, embedding_layer_weights_2d[:, 0], embedding_layer_weights_2d[:, 1]):
    plt.annotate(label, xy=(x, y),textcoords='offset points', xytext=(0, 0),size=8 )

plt.savefig('tsne_result.png')
plt.show()

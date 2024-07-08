import matplotlib.pyplot as plt
import numpy as np

SPARSITY=0.5
MODEL_NAME="mnist-net_256x"
SUBFOLDER="train_hard"

FILEPATH=f"./neuron/{SUBFOLDER}_{MODEL_NAME}"


def load_data(file_path,digit=8):

    loaded_data = np.load(file_path)
    n_layers = loaded_data['arr_0']
    n_weights = loaded_data['arr_1']
    acc = loaded_data['arr_2']
    neurons = []
    weights = []
    abs_weights = []

    for i in range(n_layers):
        neurons.append(loaded_data[f'arr_{i + 3}'])

    for i in range(n_weights):
        weights.append(loaded_data[f'arr_{i + 3 + n_layers}'])

    for i in range(n_weights):
        abs_weights.append(loaded_data[f'arr_{i + 3 + n_layers + n_weights}'])

    return acc, neurons, weights, abs_weights

def prepare_data(hidden_layer=2, sparsity=0.5, digit=8, subfolder="train_hard"):

    if digit == "none":

        acc, neurons, weights, abs_weights = load_data(f"./neuron/{subfolder}_{MODEL_NAME}{hidden_layer}_{sparsity}.npz")

    else:

        acc, neurons, weights, abs_weights = load_data(f"./neuron/{subfolder}_{MODEL_NAME}{hidden_layer}_{sparsity}_rundigit_{digit}.npz")

    weights = [array.flatten() for array in weights]
    abs_weights = [array.flatten() for array in abs_weights]

    neurons = np.concatenate(neurons)
    weights = np.concatenate(weights)
    abs_weights = np.concatenate(abs_weights)

    mask = abs_weights != 0

    result = np.where(mask, weights, np.nan)

    result_without_nan = result[~np.isnan(result)]

    return neurons

def plot_data(hidden=2, sparsity=0.0):
    hist, bin_edges = np.histogram(prepare_data(hidden, sparsity), bins=np.concatenate(np.array([0, 0.0001]), np.linspace(0.0001, 0.99, 50), np.array([0.99, 1])))

    hist = (hist) / prepare_data(hidden, sparsity).shape

    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7)

HIDDEN_LAYERS=6

arr = []

#for i in range(10):
#    hist, bin_edges = np.histogram(prepare_data(HIDDEN_LAYERS, 0, i), bins=np.linspace(0, 1, 50))
#    print(hist)
#for i in range(10):
#    arr.append(prepare_data(HIDDEN_LAYERS, 0, i))

plt.hist([prepare_data(2, 0, 10), prepare_data(2 ,0.8, 10), prepare_data(2, 0.9, 10), prepare_data(2, 0.8, "none", "sgd_trained")], bins=np.linspace(0,1,16))

#plt.hist([prepare_data(HIDDEN_LAYERS, 0.0), prepare_data(HIDDEN_LAYERS, 0.5), prepare_data(HIDDEN_LAYERS, 0.8), prepare_data(HIDDEN_LAYERS, 0.9)], bins=np.linspace(0, 1, 26), edgecolor='black', alpha=0.7, label=['Sparsity 0.0', 'Sparsity 0.5', 'Sparsity 0.8', 'Sparsity 0.9'], density=True)
#plt.scatter(prepare_data(HIDDEN_LAYERS, 0.0)[0], prepare_data(HIDDEN_LAYERS, 0.0)[1])
#plt.scatter(prepare_data(2, 0.5)[0], prepare_data(2, 0.5)[1])
#plt.scatter(prepare_data(2, 0.8)[0], prepare_data(2, 0.8)[1])
#
## Add titles and labels

#plot_data(2, 0.0)
#plot_data(2, 0.5)
#plot_data(2, 0.8)
#plot_data(2, 0.9)
#
plt.title(f'Weights activations of model with {HIDDEN_LAYERS} hidden layers (Sparse model: SGD Opt)')
plt.xlabel('Amount of weight activation across training')
plt.ylabel('Propotion out of number of parameters (%)')

#plt.title(f'Plot absolute weights vs weight activations for 2 hidden layers')
#plt.xlabel('Amount of weight activation across training')
#plt.ylabel('Absolute value of the weights')
#
## Add a legend
plt.legend()
#
## Save the plot to a PNG file
plt.show()
#plt.savefig(f'./neuron/mnist-net_256x{HIDDEN_LAYERS}_sgd_trained.png')
#

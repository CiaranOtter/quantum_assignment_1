import pennylane as qml
from pennylane import numpy as np


np.random.seed(42)
n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

dev = qml.device("lightning.qubit", wires=n_wires, shots=1)

# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])

@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))

        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

    return -objective(params), bit_strings


init = 0.01 * np.random.rand(2,1, requires_grad=True)
print(qml.draw_mpl(circuit, style="pennylane", level="device")(init[0], init[1], edge=(0,1), n_layers=1))

# perform qaoa on our graph with p=1,2 and
# keep tshe bitstring sample lists
bitstrings1 = qaoa_maxcut(n_layers=1)[1]
bitstrings2 = qaoa_maxcut(n_layers=2)[1]
    # return 
    # apply Hadamards to the given wires to achieve a superposition between the wires
    # for wire in range(n_wires):
    #     qml.Hadamard(wires=wire)

    # # p instances of unitary operators
    # for i in range(n_layers):
    #     # apply the hamiltonian with the gamma parameter
    #     U_C(gammas[i])
        
    #     # apply the mixer hamiltonia with the beta parameters
    #     U_B(betas[i])

    # if edge is None:
    #     # measurement phase
    #     return qml.sample()
    # # during the optimization phase we are evaluating a term
    # # in the objective using expval

    # H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    # return qml.expval(H)



# def qaoa_maxcut(n_layers=1):
#     print("\np={:d}".format(n_layers))

#     # initialize the parameters near zero
#     init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

#     # minimize the negative of the objective function
#     def objective(params):
#         gammas = params[0]
#         betas = params[1]
#         neg_obj = 0
#         for edge in graph:
#             # objective for the MaxCut problem
#             neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
#         return neg_obj

#     # initialize optimizer: Adagrad works well empirically
#     opt = qml.AdagradOptimizer(stepsize=0.5)

#     # optimize parameters in objective
#     params = init_params
#     steps = 30
#     for i in range(steps):
#         params = opt.step(objective, params)
#         if (i + 1) % 5 == 0:
#             print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

#     # sample measured bitstrings 100 times
#     bit_strings = []
#     n_samples = 100
#     for i in range(0, n_samples):
#         bit_strings.append(bitstring_to_int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))

#     # print optimal parameters and most frequently sampled bitstring
#     counts = np.bincount(np.array(bit_strings))
#     most_freq_bit_string = np.argmax(counts)
#     print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
#     print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

#     return -objective(params), bit_strings


# # perform qaoa on our graph with p=1,2 and
# # keep the bitstring sample lists
# bitstrings1 = qaoa_maxcut(n_layers=1)[1]
# bitstrings2 = qaoa_maxcut(n_layers=3)[1]

# xticks = range(0, 16)
# xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
# bins = np.arange(0, 17) - 0.5

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title("n_layers=1")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(bitstrings1, bins=bins)
# plt.subplot(1, 2, 2)
# plt.title("n_layers=2")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(bitstrings2, bins=bins)
# plt.tight_layout()
# plt.show()
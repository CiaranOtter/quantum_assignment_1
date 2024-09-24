import pennylane as qml
import numpy as np

def qaoa_circuit(params, edges, p):
    """
    Implements the QAOA circuit for an unweighted graph.
    
    Args:
    params (list): List of 2*p parameters [gamma1, gamma2, ..., beta1, beta2, ...]
    edges (list): List of tuples representing edges in the graph
    p (int): Number of QAOA layers
    
    Returns:
    list: Expectation values of ZZ operators for each edge
    """
    num_qubits = max(max(edge) for edge in edges) + 1
    
    # Initial state: equal superposition
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    # Alternating layers of cost and mixer unitaries
    for layer in range(p):
        # Cost unitary
        for i, j in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(-2 * params[layer], wires=j)
            qml.CNOT(wires=[i, j])
        
        # Mixer unitary
        for i in range(num_qubits):
            qml.RX(2 * params[p + layer], wires=i)
    
    # Return expectation values for each edge
    return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in edges]

def cost_function(params, edges, p):
    """
    Computes the cost for the QAOA optimization.
    
    Args:
    params (list): List of 2*p parameters [gamma1, gamma2, ..., beta1, beta2, ...]
    edges (list): List of tuples representing edges in the graph
    p (int): Number of QAOA layers
    
    Returns:
    float: The cost (negative of the objective function to be maximized)
    """
    expectation_values = qaoa_circuit(params, edges, p)
    # Convert expectation values to a numpy array and sum them
    return -np.sum(np.array(expectation_values))

def optimize_qaoa(edges, p, steps=100):
    """
    Optimizes the QAOA circuit for an unweighted graph.
    
    Args:
    edges (list): List of tuples representing edges in the graph
    p (int): Number of QAOA layers
    steps (int): Number of optimization steps
    
    Returns:
    array: Optimized parameters
    float: Final cost
    """
    num_qubits = max(max(edge) for edge in edges) + 1
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def circuit(params):
        return qaoa_circuit(params, edges, p)
    
    # Initial random parameters
    init_params = np.random.uniform(0, np.pi, 2*p)
    
    # Optimization
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    params = init_params
    
    for i in range(steps):
        params = opt.step(lambda params: cost_function(params, edges, p), params)
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: cost = {cost_function(params, edges, p):.4f}")
    
    return params, cost_function(params, edges, p)

# Example usage
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]  # Example unweighted graph
p = 2  # Number of QAOA layers

optimal_params, final_cost = optimize_qaoa(edges, p)
print(f"Optimal parameters: {optimal_params}")
print(f"Final cost: {final_cost}")
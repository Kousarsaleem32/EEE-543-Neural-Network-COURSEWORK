import cupy as cp
import h5py
# Load data
with h5py.File('data-Mini Project 2.h5', 'r') as data:
    # print(file['trX'])
    trXX, trYY, tstXX, tstYY = data['trX'], data['trY'], data['tstX'], data['tstY']
    trX = trXX[:]
    trY = trYY[:]
    tstX = tstXX[:]
    tstY = tstYY[:]

# print(trX, trY, tstX, tstY)

# Hyperparameters
learning_rates = [0.05, 0.1]
N_values = [50, 100]
batch_sizes = [10, 30]
epochs = 10

# Activation functions
def tanh(x):
    return cp.tanh(x)

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


# Initialize weights and biases
def initialize_parameters(N):
    cp.random.seed(42)
    WIH = cp.random.uniform(-0.1, 0.1, size=(N, 4))
    WHH = cp.random.uniform(-0.1, 0.1, size=(N, N))
    WHO = cp.random.uniform(-0.1, 0.1, size=(6, N + 1))
    return WIH, WHH, WHO

# One-hot encoding
def one_hot_encode(labels):
    labels = labels.astype(int)  # Convert labels to integer type
    one_hot = cp.zeros((len(labels), 6))
    one_hot[cp.arange(len(labels)), labels-1] = 1
    return one_hot


# Forward pass
def forward_pass(inputs, WIH, WHH, WHO):
    hidden_states = [cp.zeros((inputs.shape[0], WIH.shape[0]))]
    for t in range(inputs.shape[1]):
        x_t = cp.hstack([inputs[:, t, :], cp.ones((inputs.shape[0], 1))])
        h_t = tanh(cp.dot(x_t, WIH.T) + cp.dot(hidden_states[-1], WHH.T))
        hidden_states.append(h_t)

    output = sigmoid(cp.dot(cp.hstack([hidden_states[-1], cp.ones((inputs.shape[0], 1))]), WHO.T))
    return hidden_states, output

# Backward pass
def backward_pass(inputs, labels, hidden_states, output, WIH, WHH, WHO, learning_rate):
    error = output - labels
    dWHO = cp.dot(error.T, cp.hstack([hidden_states[-1], cp.ones((inputs.shape[0], 1))]))
    
    dWHH = cp.zeros_like(WHH)
    dWIH = cp.zeros_like(WIH)
    delta = cp.dot(error, WHO[:, :-1]) * (1 - hidden_states[-1]**2)

    for t in range(inputs.shape[1], 0, -1):
        x_t = cp.hstack([inputs[:, t-1, :], cp.ones((inputs.shape[0], 1))])
        dWHH += cp.dot(delta.T, hidden_states[t-1])
        dWIH += cp.dot(delta.T, x_t)
        delta = cp.dot(delta, WHH) * (1 - hidden_states[t-1]**2)
    # Update weights and biases
    WHO -= learning_rate * dWHO
    WHH -= learning_rate * dWHH
    WIH -= learning_rate * dWIH

    return WIH, WHH, WHO

# Training loop
for learning_rate in learning_rates:
    for N in N_values:
        for batch_size in batch_sizes:
            WIH, WHH, WHO = initialize_parameters(N)

            # Training
            for epoch in range(epochs):
                for i in range(0, trX.shape[0], batch_size):
                    batch_inputs = trX[i:i+batch_size]
                    batch_labels = cp.asarray(trY[i:i+batch_size])

                    hidden_states, output = forward_pass(batch_inputs, WIH, WHH, WHO)
                    WIH, WHH, WHO = backward_pass(batch_inputs, batch_labels, hidden_states, output, WIH, WHH, WHO, learning_rate)

                # Validation error (not part of the training loop)
                val_error = 0
                for j in range(0, 3000, 50):
                    val_inputs = trX[j:j+50]
                    val_labels = cp.asarray(trY[j:j+50])
                    _, val_output = forward_pass(val_inputs, WIH, WHH, WHO)
                    val_error += cp.sum(-val_labels * cp.log(val_output + 1e-10))

                print(f"Epoch {epoch+1}/{epochs}, Validation Error: {val_error/3000}")

            # Testing
            _, test_output = forward_pass(tstX, WIH, WHH, WHO)
            test_predictions = cp.argmax(test_output, axis=1)
            test_accuracy = cp.mean(test_predictions + 1 == cp.argmax(cp.asarray(tstY), axis=1))
            print(f"Learning Rate: {learning_rate}, N: {N}, Batch Size: {batch_size}")
            print("Training Error:", val_error/3000)
            print("Test Accuracy:", test_accuracy)
            print()



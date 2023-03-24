import matplotlib.pyplot as plt

with open('vals.txt') as f:
    lines = f.readlines()[::2]
    loss = []
    for line in lines:
        a, b = line.split('=')[1][2:].split()[0].split('+')
        a = a[:-1]
        num = float(a) * (10 ** float(b))
        loss.append(num)
    plt.plot([x * 10 for x in range(1, len(loss) + 1)], loss)
    plt.yscale('log')
    plt.title('Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Log)')
    plt.show()

class NN:
    def __init__(self, input_size, output_size, epochs=10, alpha=0.1, batch_size=0, Regularizer=0.0):
        self.layers = [input_size, 200, 80, output_size]
        self.activation_funcs = ['relu', 'sigmoid', 'softmax']
        self.inputs = [epochs, alpha, batch_size, Regularizer]
        self.epochs = epochs
        self.alpha = alpha
        self.Reg = Regularizer
        self.batch_size = batch_size
        self.Thetas = self.initModel()

    def initModel(self):
        Thetas = {}
        for layer in range(len(self.layers) - 1):
            Thetas[f'T{layer + 1}'] = np.random.randn(self.layers[layer] + 1, self.layers[layer + 1]) / 10
        return Thetas

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def dxsigmoid(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def softmax(self, x):
        x = np.array(x)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1)[:, None]

    def Relu(self, x):
        return np.maximum(0, x)

    def dxRelu(self, x):
        return np.where(x >= 0, 1, 0)

    def forward(self, X):
        X = np.matrix(X)
        m = X.shape[0]
        att = self.activation_funcs
        Thetas = self.Thetas
        forward_steps = {'a1': X}
        Ultimo_layer = int(len(self.layers))
        for layer in range(1, Ultimo_layer):
            forward_steps[f'z{layer + 1}'] = np.dot(forward_steps[f'a{layer}'], Thetas[f'T{layer}'])
            if att[layer - 1] == 'sigmoid':
                forward_steps[f'a{layer + 1}'] = np.concatenate(
                    (np.ones([m, 1]), self.sigmoid(forward_steps[f'z{layer + 1}'])), axis=1)
            elif att[layer - 1] == 'softmax':
                forward_steps[f'a{layer + 1}'] = np.concatenate(
                    (np.ones([m, 1]), self.softmax(forward_steps[f'z{layer + 1}'])), axis=1)
            elif att[layer - 1] == 'relu':
                forward_steps[f'a{layer + 1}'] = np.concatenate(
                    (np.ones([m, 1]), self.Relu(forward_steps[f'z{layer + 1}'])), axis=1)

        h = forward_steps.pop(f'a{Ultimo_layer}')
        forward_steps['h'] = h[:, 1:]
        return forward_steps

    def loss(self):
        Y = self.Y
        X = self.X
        Thetas = self.Thetas
        m = self.m
        Reg = self.Reg
        soma_weights = 0
        for i in range(len(Thetas)):
            weights = Thetas[f'T{i + 1}']
            weights[0] = 0
            soma_weights += np.sum(weights ** 2)
        Forward_dict = self.forward(X)
        h = Forward_dict['h']
        soma = np.sum((np.multiply(-Y, np.log(h)) - np.multiply((1 - Y), (np.log(1 - h)))))
        J = soma / m + (Reg / (2 * m)) * soma_weights
        return J

    def grad_calc(self, X, Y):
        X = np.matrix(X)
        Y = np.matrix(Y)
        m = X.shape[0]
        Thetas = self.Thetas
        n_layers = len(self.layers)
        att = self.activation_funcs
        Thetas_grad = []

        Forward_list = self.forward(X)
        deltas = {}
        deltas[f'delta{n_layers}'] = Forward_list['h'] - Y
        for i in range(n_layers - 1, 1, -1):
            if att[i - 2] == 'sigmoid':
                deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i + 1}'], Thetas[f'T{i}'][1:].T)),
                                                  self.dxsigmoid(Forward_list[f'z{i}']))
            elif att[i - 2] == 'relu':
                deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i + 1}'], Thetas[f'T{i}'][1:].T)),
                                                  self.dxRelu(Forward_list[f'z{i}']))

        for c in range(len(deltas)):
            BigDelta = np.array(np.dot(deltas[f'delta{c + 2}'].T, Forward_list[f'a{c + 1}']))
            weights = Thetas[f'T{c + 1}']
            weights[0] = 0
            grad = np.array(BigDelta + (self.Reg * weights.T)) / m
            Thetas_grad.append(grad)
        return Thetas_grad

    def accuracy_calc(self, X, Y):
        Forward_list = self.forward(X)
        h = Forward_list['h']
        y_hat = np.argmax(h, axis=1)[:, None]
        y = np.argmax(Y, axis=1)[:, None]
        return np.mean(y_hat == y)

    def train(self, X, Y, x_test, y_test):
        Thetas = self.Thetas
        self.X = X
        self.Y = Y
        self.m = X.shape[0]
        j_history = []
        train_acc, test_acc = [], []
        if self.batch_size <= 0:
            b_size = self.m
        elif isinstance(self.batch_size, int) and (1 <= self.batch_size <= self.m):
            b_size = self.batch_size
        else:
            return
        for ep in range(self.epochs):
            m = self.m
            a = np.array([0, b_size])
            num = 1

            for i in range(m // b_size):
                inx = a + b_size * i
                grad_list = self.grad_calc(X[inx[0]:inx[1]], Y[inx[0]:inx[1]])
                for g in range(len(grad_list)):
                    self.Thetas[f'T{g + 1}'] = self.Thetas[f'T{g + 1}'] - self.alpha * np.array(grad_list[g]).T

            if (ep + 1) % num == 0:
                J = self.loss()
                j_history.append(J)
                accu_train = self.accuracy_calc(X, Y)
                accu_test = self.accuracy_calc(x_test, y_test)
                test_acc.append(accu_train)
                train_acc.append(accu_test)
                print(
                    f'Epoch: {ep + 1} | Loss: {J:.5f}: Train Accuracy: {accu_train:.5%} | Test Accuracy: {accu_test:.5%}')
        return j_history, train_acc, test_acc
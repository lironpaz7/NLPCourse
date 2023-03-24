import os
import warnings
import torch.utils.data as data_utils
from gensim import downloader
from utils import read_data_model2, down_sample
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings(action='ignore')


class Q2NN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(Q2NN, self).__init__()
        dims = [input_size, 64, 32, 16, 32, 64, 32, 16, output_size]
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(in_features=dims[i - 1], out_features=dims[i]))
            layers.append(nn.ReLU())
        layers.pop()  # removes the last ReLU as it is not necessary
        layers.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # running the model linear layers
        return self.sequential(x)


def train(*, model, train_path, loss_fn, optimizer, device, epochs, embeds):
    X_train, y_train = read_data_model2(train_path, embeds)
    X_train, y_train = down_sample(X_train, y_train, biggest_class=0, leave=30000)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    train = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
    loss_lst = []
    for epoch in tqdm(range(epochs), desc='Training Model...', ncols=100):
        epoch_loss = 0
        for j, (x_train, y_train) in enumerate(train_loader):
            output = model(x_train)
            loss = loss_fn(output, y_train.reshape(-1, 1))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = round(epoch_loss / len(train_loader), 3)
        loss_lst.append(epoch_loss)
        # print(f'[{epoch + 1}/{epochs}]{epoch_loss}')
    with open('loss.txt', 'w') as f:
        f.write(' '.join([str(x) for x in loss_lst]))


def run_model(train_path, test_path, n=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device: {device}')

    model = Q2NN(input_size=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    loss_fn = nn.BCELoss()

    if not os.path.exists('embeds.pkl'):
        print('Downloading glove embedding model...')
        embeds = downloader.load(f"glove-twitter-{n}")
        torch.save(embeds, 'embeds.pkl')
        print('Finished downloading glove embedding model...')
    else:
        print('Loading glove embedding model...')
        embeds = torch.load('embeds.pkl')

    train(model=model, train_path=train_path, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=30,
          embeds=embeds)
    torch.save(model, 'weights.pkl')

    X_test, y_test = read_data_model2(test_path, embeds)
    X_test = X_test.to(device)
    y_test = y_test.detach().numpy()
    print('Predicting...')
    y_pred = model(X_test).cpu()
    y_pred = torch.round(y_pred).detach().numpy()
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'F1 Score: {f1 :.3f}')


if __name__ == "__main__":
    train_path = "data/train.tagged"
    test_path = "data/dev.tagged"
    print(f'Running model evaluation with train path={train_path} and test path={test_path}...')
    run_model(train_path, test_path)

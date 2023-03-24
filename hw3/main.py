from hw3.model import DepParserLA
from hw3.train import train_model
from utils import *
from torch.utils.data.dataloader import DataLoader
import torch
from torch import nn, optim
from constants import *

torch.manual_seed(28)
torch.cuda.manual_seed(28)
random.seed(28)
np.random.seed(28)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    files = {'train': "data/train.labeled", 'test': "data/test.labeled"}
    print('Getting vocabs...')
    word_vocabulary, pos_vocabulary = get_vocabs(files)
    train_dataset = PosDataset(files['train'], word_vocabulary, pos_vocabulary)
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataset = PosDataset(files['test'], word_vocabulary, pos_vocabulary)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    word_vocab_size = len(train_dataset.word_idx_mappings)
    tag_vocab_size = len(train_dataset.pos_idx_mappings)
    print('Getting embeddings from glove...')
    glove_embeddings = get_embeddings_from_glove(word_vocabulary)
    print('Creating model...')
    model = DepParserLA(pos_embedding_dim=POS_EMBEDDING_DIM,
                        hidden_dim=HIDDEN_DIM, word_vocab_size=word_vocab_size, tag_vocab_size=tag_vocab_size,
                        glove_embedding=glove_embeddings)

    if torch.cuda.is_available():
        model.cuda()

    loss_function = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, device, train_dataloader, test_dataloader, EPOCHS, optimizer, loss_function,
                GRAD_ACC, plot=True)


if __name__ == "__main__":
    print('Liron & Adir')
    run()

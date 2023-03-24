import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import chu_liu_edmonds
from hw3.constants import *
from main import DepParserLA
from utils import get_vocabs, PosDataset, get_embeddings_from_glove, CompDataset


def tag_comp(model, device, loader, file_name='comp.labeled'):
    results = []
    for _, input_data in enumerate(tqdm(loader, desc='Tagging...', ncols=100)):
        words_idx, pos_idx, sentence = input_data
        words_idx_tensor = torch.tensor(words_idx, dtype=torch.long).to(device)
        pos_idx_tensor = torch.tensor(pos_idx, dtype=torch.long).to(device)
        sentence_length = len(sentence)
        edges_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
        x = np.array(torch.detach(edges_scores).to("cpu"))
        mst_tree, _ = chu_liu_edmonds.decode_mst(x, len(x), has_labels=False)
        mst_tree = np.delete(mst_tree, 0)
        counter = 1
        sentence.remove(('ROOT', 'ROOT'))
        for (word, pos), head in zip(sentence, mst_tree):
            line = [str(counter), word, '_', pos, '_', '_', str(head), '_', '_', '_\n']
            results.append("\t".join(line))
            counter += 1
        results.append("\n")
    with open(file_name, "w") as f:
        f.writelines(results)
    print(f'{file_name} written successfully!')


def eval_on_comp(path_to_comp="data/comp.unlabeled", chosen_model='model_epoch_20.pkl'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    files = {'train': "data/train.labeled", 'test': "data/test.labeled"}

    print('Getting vocabs...')
    word_vocabulary, pos_vocabulary = get_vocabs(files)
    train_dataset = PosDataset(files['train'], word_vocabulary, pos_vocabulary)
    word_idx_mappings = train_dataset.word_idx_mappings
    pos_idx_mappings = train_dataset.pos_idx_mappings
    word_vocab_size, pos_vocab_size = len(word_idx_mappings), len(pos_idx_mappings)

    print('Creating model...')
    glove_embeddings = get_embeddings_from_glove(word_vocabulary)
    model = DepParserLA(pos_embedding_dim=POS_EMBEDDING_DIM,
                        hidden_dim=HIDDEN_DIM, word_vocab_size=word_vocab_size, tag_vocab_size=pos_vocab_size,
                        glove_embedding=glove_embeddings)

    print(f'Loading {chosen_model} file...')
    model.load_state_dict(torch.load(chosen_model))
    model.to(device)
    print('Set eval mode...')
    model.eval()

    print('Loading competition dataset...')
    comp_dataset = CompDataset(file_path=path_to_comp, word_idx_mapping=word_idx_mappings,
                               pos_idx_mapping=pos_idx_mappings)
    comp_dataloader = DataLoader(comp_dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x[0])

    tag_comp(model, device, comp_dataloader)


if __name__ == '__main__':
    eval_on_comp(path_to_comp="data/comp.unlabeled", chosen_model='model_epoch_20.pkl')

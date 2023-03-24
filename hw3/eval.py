import numpy as np
import torch
from torch import nn

from hw3 import chu_liu_edmonds


def evaluate(dataloader, model):
    running_acc, len_words, running_loss = 0, 0, 0
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_function = nn.NLLLoss().to(device)
        for _, inp in enumerate(dataloader):
            words_idx_tensor, pos_idx_tensor, sent_len, headers = inp
            edges_scores = model(words_idx_tensor, pos_idx_tensor, sent_len)
            log_softmax_scores = torch.nn.functional.log_softmax(edges_scores, 0)
            running_loss += loss_function(log_softmax_scores.t()[1:, :], torch.tensor(headers, device=device)).item()
            x = np.array(torch.detach(edges_scores).to("cpu"))
            mst_tree, _ = chu_liu_edmonds.decode_mst(x, len(x), has_labels=False)
            mst_tree = np.delete(mst_tree, 0)
            running_acc += np.sum(mst_tree == np.array(torch.stack(headers).squeeze()))
            len_words += sent_len.item()
        acc = running_acc / len_words
        running_loss = running_loss / len_words
    return acc, running_loss

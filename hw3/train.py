import time

import torch

from hw3.eval import evaluate
from hw3.utils import plot_graphs


def train_model(model, device, train_loader, test_loader, epochs, optimizer, criterion, grads_acc,
                plot=True):
    print('-' * 50)
    print('Training...')
    s = time.time()
    stats = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    for epoch in range(epochs):
        for batch_idx, input_data in enumerate(train_loader, start=1):
            words_idx_tensor, pos_idx_tensor, sentence_length, headers = input_data
            edges_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
            headers = torch.tensor(headers, device=device)
            log_softmax_scores = torch.nn.functional.log_softmax(edges_scores, 0)
            loss = criterion(log_softmax_scores.t()[1:, :], headers)
            loss = loss / grads_acc
            loss.backward()
            if batch_idx % grads_acc == 0:
                optimizer.step()
                model.zero_grad()
        train_acc, train_loss = evaluate(train_loader, model)
        test_acc, test_loss = evaluate(test_loader, model)
        print(
            f"[{epoch + 1}/{epochs}] | Train Loss: {train_loss: .4f} | Train UAS: {train_acc: .4f} | Test Loss: {test_loss : .4f} | Test UAS: {test_acc: .4f}")
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['test_loss'].append(test_loss)
        stats['test_acc'].append(test_acc)
        model_epoch_to_save = f"model_epoch_{epoch + 1}.pkl"
        print(f'Saving model checkpoint: {model_epoch_to_save}...')
        torch.save(model.state_dict(), model_epoch_to_save)
        print('-' * 90)
    print('Done!')
    print(f'Total train time: {time.time() - s}')

    print('Done!')
    if plot:
        plot_graphs(stats)

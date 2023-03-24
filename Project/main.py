from typing import Optional
import torch

from generate_comp_tagged import tag_comp
from train import train_model

torch.manual_seed(28)
torch.cuda.manual_seed(28)


def run(train_path: str = './data/train.labeled', val_unlabeled_path: str = './data/val.unlabeled',
        val_labeled_path: str = './data/val.labeled', weights: str = None, limit: Optional[int] = None):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device}...')
    print('Loading Model & Tokenizer...')
    model, tokenizer = train_model(device=device, train_file=train_path, validation_file=val_labeled_path,
                                   weights=weights)
    print('Done!')
    tag_comp(model=model,
             tokenizer=tokenizer,
             device=device,
             untagged_file=val_unlabeled_path,
             tagged_name='./data/val.labeled',
             tagged_file=val_labeled_path)


if __name__ == "__main__":
    print('Liron & Adir')
    # run(weights='./weights.pkl')
    run()

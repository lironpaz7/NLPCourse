import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from eval import translate_all
from project_evaluate import calculate_score
from utils import construct_paragraph


def tag_comp(model, tokenizer, device: str, untagged_file: str = 'val.unlabeled',
             tagged_name: str = 'comp.labeled',
             tagged_file: str = None) -> None:
    """
    Tags the untagged file with the given model and tokenizer and produces tagged_name file.
    :param model: Model
    :param tokenizer: Tokenizer
    :param device: cuda or cpu
    :param untagged_file: The untagged file path
    :param tagged_name: The file name which will be created
    :param tagged_file: The tagged file path. Default: None
    """
    with open(untagged_file, encoding='utf-8') as f:
        with open(tagged_name, "w", encoding='utf-8') as w:
            de_texts = []
            for line in tqdm(f.readlines(), desc='Translating...', ncols=100):
                line = line.strip().replace('\n', '')
                if line == 'German:':
                    if de_texts:
                        predicted_texts = translate_all(model, tokenizer, device, de_texts, apply_tqdm=False)
                        results = construct_paragraph(de_texts=de_texts, en_texts=predicted_texts)
                        w.writelines(results)
                        w.write('\n')
                        de_texts = []
                elif line.startswith('Roots') or line.startswith('Modifiers') or line == '':
                    continue
                else:
                    de_texts.append(line)
            if de_texts:
                predicted_texts = translate_all(model, tokenizer, device, de_texts, apply_tqdm=False)
                results = construct_paragraph(de_texts=de_texts, en_texts=predicted_texts)
                w.writelines(results)
    if tagged_file:
        calculate_score(file_path1=tagged_name, file_path2=tagged_file)

    print(f'{tagged_name} written successfully!')


def run_generate_val(model, tokenizer, device, untagged_file='./data/val.unlabeled',
                     tagged_name='val.labeled'):
    """
    Runs the tagger model on the given untagged file and creates the tagged_name file
    :param model: Tagger model
    :param tokenizer: Tokenizer
    :param device: cuda or cpu
    :param untagged_file: The untagged file that the model will tag
    :param tagged_name: The tagged file that the model will create
    """
    tag_comp(model=model,
             tokenizer=tokenizer,
             device=device,
             untagged_file=untagged_file,
             tagged_name=tagged_name)


def run_generate_comp(model, tokenizer, device, untagged_file='./data/comp.unlabeled',
                      tagged_name='comp.labeled'):
    """
    Runs the tagger model on the given untagged file and creates the tagged_name file
    :param model: Tagger model
    :param tokenizer: Tokenizer
    :param device: cuda or cpu
    :param untagged_file: The untagged file that the model will tag
    :param tagged_name: The tagged file that the model will create
    """
    tag_comp(model=model,
             tokenizer=tokenizer,
             device=device,
             untagged_file=untagged_file,
             tagged_name=tagged_name)


if __name__ == '__main__':
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("t5-base", padding='max_length', truncation=True)
    checkpoint = 'checkpoint-7000'
    model = f'./base-t5-liron-adir-de-to-en/{checkpoint}'
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    # print('Running generate on val file:')
    # run_generate_val(model=model, tokenizer=tokenizer, device=device)

    print('-' * 100)

    print('Running generate on comp file:')
    run_generate_comp(model=model, tokenizer=tokenizer, device=device)

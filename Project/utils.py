from datasets import DatasetDict, Dataset
import spacy
from tqdm import tqdm

from constants import *

roots_modifiers_model = spacy.load("en_core_web_sm")


def construct_paragraph(de_texts: Texts, en_texts: Texts) -> Texts:
    """
    Constructs a paragraph from the given German texts and English texts
    :param de_texts: List of German texts
    :param en_texts: List of English texts
    :return: Paragraph containing both German and English texts
    """
    paragraph = ['German:\n']
    # German texts
    for t in de_texts:
        paragraph.append(t + '\n')

    # English texts
    paragraph.append('English:\n')
    for t in en_texts:
        paragraph.append(t + '\n')

    return paragraph


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def get_translations(translator, de_texts: Texts) -> Texts:
    """
    Extract the translations from the German texts using the translator model
    :param translator: Translator model
    :param de_texts: List of German texts
    :return: English texts translated from German
    """
    trans = translator(de_texts, max_length=750)
    trans = [t['translation_text'] for t in trans]
    return trans


def convert_text_to_list(text: str) -> list:
    text = text.split('\n')
    lst = []
    for s in text:
        s = s.strip()
        if s.startswith('Roots') or s.startswith('Modifiers') or s == '':
            continue
        lst.append(s + '\n')
    return lst


def get_roots_and_modifiers(d: dict) -> None:
    for p_dict in tqdm(d['translation'], desc='Analyzing...', ncols=100):
        de_texts = p_dict['de'].split('\n')
        en_texts = p_dict['en'].split('\n')
        roots, modifiers = [], []
        for s_de, s_en in zip(de_texts, en_texts):
            s_de = s_de.strip()
            s_en = s_en.strip()
            if s_de == '':
                continue
            doc = roots_modifiers_model(s_en)
            for token in doc:
                if token.head == token:
                    roots.append(token.text)
                    mods = [child.text for child in token.children if child.dep_ != 'punct'][:2]
                    modifiers.append(mods)

        # add roots and modifiers to paragraph
        add_ = fix_format(roots, modifiers)
        de_texts = '\n'.join(de_texts) + add_
        p_dict['de'] = de_texts


def fix_format(roots, modifiers) -> str:
    """
    Prepare the roots & modifiers
    :param roots:
    :param modifiers:
    :return:
    """
    roots_fixed = ', '.join([elem for elem in roots])
    modifiers_fixed = ', '.join([f"({', '.join([elem for elem in modifiers])})" for modifiers in modifiers])
    return f"Roots in English: {roots_fixed}\nModifiers in English: {modifiers_fixed}\n"


def dataset_helper(train_path: str, add_modifiers_and_root: bool = False) -> dict:
    """
    Construct the dataset dict from the given train_path file. Adds modifiers and root if asked.
    :param train_path: Path to the train file
    :param add_modifiers_and_root:
    :return:
    """
    with open(train_path) as f:
        d = {'translation': list()}
        tmp, lang = {}, ''
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            if line == 'German:':
                lang = 'de'
                tmp[lang] = ''
            elif line == 'English:':
                lang = 'en'
                tmp[lang] = ''
            elif line != '':
                tmp[lang] += f'{line}\n'
            else:
                d['translation'].append(tmp)
                tmp = dict()
        if add_modifiers_and_root:
            print('Extracting roots and modifiers...')
            get_roots_and_modifiers(d)
            print('Done!')
        return d


def create_dataset(train_file: str, val_file: str, tokenizer, test_size: float = 0.2) -> DatasetDict:
    """
    Creates a dataset from the given files in the format of a dictionary contains train and validation
    :param train_file: Path to the train file
    :param val_file: Path to the val file
    :param tokenizer: Tokenizer
    :param test_size: The size of the test in the split method
    :return: DatasetDict Object
    """
    dataset_train = dataset_helper(train_file)
    dataset_validation = dataset_helper(val_file)
    datasets = DatasetDict({
        'train': Dataset.from_dict(dataset_train),
        'validation': Dataset.from_dict(dataset_validation)
    })

    def tokenizer_helper(examples, prefix='translate German to English: '):
        inputs = [prefix + ex[SOURCE_LANGUAGE] for ex in examples["translation"]]
        targets = [ex[TARGET_LANGUAGE] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return datasets.map(tokenizer_helper, batched=True)

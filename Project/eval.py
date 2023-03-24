from tqdm import tqdm
from constants import Texts
from project_evaluate import compute_metrics


def translate(model, tokenizer, device, text: str) -> str:
    """
    Translate a single sentence with the given model and tokenizer.
    :param device: cuda or cpu
    :param model: Model that will generate the prediction
    :param tokenizer: Tokenizer
    :param text: Sentence to translate (in German)
    :return: The predicted translation for this sentence
    """
    # Tokenize text
    tokenized_text = tokenizer(text, return_tensors='pt').to(device).input_ids
    # TODO: Check the max_length parameter
    translation = model.generate(tokenized_text, max_length=750)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    return translated_text


def translate_all(model, tokenizer, device, de_texts: Texts, true_en: Texts = None, score=True,
                  apply_tqdm=True) -> Texts:
    """
    Translates all the given sentences from German to English.
    :param device: cuda or cpu
    :param apply_tqdm: Applies the tqdm loading bar. Default: True
    :param model: Model that will generate the prediction
    :param tokenizer: Tokenizer
    :param de_texts: List of texts in the source language (German)
    :param true_en: List of translated texts in the target language (English)
    :param score: If set to true, calculate the bleu score. Default: True
    :return: List of predicted sentences
    """
    tagged_en = []
    if apply_tqdm:
        for text in tqdm(de_texts, desc='Translating...', ncols=100):
            en_txt = translate(model, tokenizer, device, text)
            tagged_en.append(en_txt)
    else:
        for text in de_texts:
            en_txt = translate(model, tokenizer, device, text)
            tagged_en.append(en_txt)
    if score and true_en:
        bleu_score = compute_metrics(tagged_en, true_en)
        print(f'Bleu Score: {bleu_score}')
    return tagged_en

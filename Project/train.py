import pickle
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoTokenizer
from constants import *
from utils import create_dataset, postprocess_text


def train_model(device: str = 'cuda', train_file: str = './data/train.labeled',
                validation_file: str = './data/val.labeled', weights: str = None):
    """
    Trains the t5-base model with the given train data
    :param validation_file: Path to the validation file
    :param device: cuda or cpu. Default: cuda
    :param train_file: Path to the train data
    :param weights: Weights form pretrained session. Default: None
    :return: The trained model
    """
    tokenizer = AutoTokenizer.from_pretrained("t5-base", padding='max_length', truncation=True)
    print("Loading base-t5 model...")
    if weights:
        print('Loading pretrained model...')
        with open(weights, 'rb') as f:
            model = pickle.load(f)
        return model.to(device), tokenizer
    # --------------------------------------------------------------

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    print(f"Loading data from {train_file}...")
    datasets = create_dataset(train_file=train_file, val_file=validation_file, tokenizer=tokenizer, test_size=0.2)
    metric = load_metric("sacrebleu")

    def compute_metrics_eval(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    model_trainer = Seq2SeqTrainer(
        model=model,
        args=TRAINER_ARGS,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_eval
    )
    print('Training...')
    model_trainer.train()
    with open('weights.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model, tokenizer

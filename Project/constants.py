from typing import List

from transformers import Seq2SeqTrainingArguments

SOURCE_LANGUAGE = "de"
TARGET_LANGUAGE = "en"
Texts = List[str]
TRAINER_ARGS = Seq2SeqTrainingArguments(
    f"base-t5-liron-adir-de-to-en_roots",
    learning_rate=0.0004,
    evaluation_strategy='epoch',
    weight_decay=0.02,
    save_total_limit=4,
    fp16=True,
    adafactor=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    generation_max_length=220,
    predict_with_generate=True
)

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


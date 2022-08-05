from pyrsistent import T
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pickle
import os

os.environ['WANDB_DISABLED'] = 'true'
model_name = 'indolem/indobert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tok = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

with open('spc_sum_data.pck','rb') as f:
    data = pickle.load(f)

# data expected is datasetdict with train and test with each having (teks,kalimat,label)

def tokenize_f(examples):
    tokenized_inputs = tok(
        examples["sentence1"], examples['sentence2'], truncation=True, padding="max_length"
    )

    tokenized_inputs['labels'] = examples['label']

    return tokenized_inputs

tokenized_ds = data.map(tokenize_f)

args = TrainingArguments(
    "Indobert-SPC-ExtSum",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    weight_decay=0.01
#     push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds['test'],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
    tokenizer=tok,
)

trainer.train()
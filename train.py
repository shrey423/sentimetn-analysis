import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import get_scheduler
import numpy as np
from torch.optim import AdamW
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from CustomDataset import CustomDataset  


nltk.download('stopwords')
nltk.download('wordnet')


with open('primate_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


for entry in data:
    entry['post_text'] = preprocess_text(entry['post_text'])

train_data, remaining_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)


model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(data[0]["annotations"]))

train_dataset = CustomDataset(train_data, tokenizer, max_length=512)
val_dataset = CustomDataset(val_data, tokenizer, max_length=512)
test_dataset = CustomDataset(test_data, tokenizer, max_length=512)



train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


def custom_data_collator(batch):
    inputs = {key: torch.stack([sample[key] for sample in batch]) for key in batch[0].keys() if key != 'labels'}
    inputs['labels'] = torch.stack([sample['labels'] for sample in batch])
    return inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * 3,
)
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)
 
def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    pred_probs = 1 / (1 + np.exp(-predictions)) 
    pred_labels = (pred_probs >= 0.5).astype(int)

    accuracy=Accuracy(labels, pred_labels)
    precision = precision_score(labels, pred_labels, average='macro')
    recall = recall_score(labels, pred_labels, average='macro')
    f1 = f1_score(labels, pred_labels, average='macro')
    roc_auc = roc_auc_score(labels, pred_probs, average='macro', multi_class='ovr')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
    }

training_args.compute_metrics = compute_metrics

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./bert_base_cased_model')


test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test set results:", test_results)

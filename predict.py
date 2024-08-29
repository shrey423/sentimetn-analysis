import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

with open('primate_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

model = BertForSequenceClassification.from_pretrained('./bert_base_cased_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

with open('test.txt', 'r', encoding='utf-8') as test_file:
    new_post_text = test_file.read()


inputs = tokenizer(new_post_text, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)


threshold = 0.65
predicted_labels = ["yes" if pred >= threshold else "no" for pred in predictions[0]]

result_structure = [
    [label[0], pred_label] for label, pred_label in zip(data[0]["annotations"], predicted_labels)
]

print("Predicted Labels:")
for label, pred_label in result_structure:
    print(f"{label}: {pred_label}")

with open('result.json', 'w', encoding='utf-8') as result_file:
    json.dump(result_structure, result_file, ensure_ascii=False, indent=2)

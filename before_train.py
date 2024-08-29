import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=10)

model.eval()

label_map = {
    0: "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down",
    1: "Feeling-down-depressed-or-hopeless",
    2: "Feeling-tired-or-having-little-energy",
    3: "Little-interest-or-pleasure-in-doing",
    4: "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual",
    5: "Poor-appetite-or-overeating",
    6: "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way",
    7: "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television",
    8: "Trouble-falling-or-staying-asleep-or-sleeping-too-much",
    9: "Other"
}

def predict_labels(text):
 
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities >= 0.5).int() 

    result = {label_map[i]: ('yes' if predictions[0][i].item() == 1 else 'no') for i in range(len(label_map))}
    return result

if __name__ == "__main__":
    text = "I want to die, I am depressed."
    predictions = predict_labels(text)

    print("Predicted Labels:")
    for label, pred in predictions.items():
        print(f"{label}: {pred}")

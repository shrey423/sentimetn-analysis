import json
import matplotlib.pyplot as plt
import numpy as np


with open('primate_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


annotations = [entry["annotations"] for entry in data]


short_names = {
    "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down": "Bad",
    "Feeling-down-depressed-or-hopeless": "Depressed",
    "Feeling-tired-or-having-little-energy": "Tired",
    "Little-interest-or-pleasure-in-doing ": "LittleInterest",
    "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual": "Restless",
    "Poor-appetite-or-overeating": "Appetite",
    "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way": "Thoughts",
    "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television": "Concentration",
    "Trouble-falling-or-staying-asleep-or-sleeping-too-much": "SleepTrouble"
}


counts_dict = {short_names[label]: {"yes": 0, "no": 0} for label in short_names}


for entry in annotations:
    for label, value in entry:
        counts_dict[short_names[label]][value] += 1

fig, ax = plt.subplots(figsize=(12, 8))


for label, counts in counts_dict.items():
    ax.bar([f"{label}_yes", f"{label}_no"], [counts["yes"], counts["no"]])

ax.set_ylabel('Count')
ax.set_title('Distribution of Annotations')
ax.legend(["Yes", "No"])  
plt.xticks(rotation=45, ha="right")

plt.show()

from prefix_tree import N_gram
import statistics
import os
import json
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    unigram = N_gram(1)
    bigram = N_gram(2)
    trigram = N_gram(3)
    five_gram = N_gram(5)

    unigram.load_from_file("./saves/model/unigram_prefix_save")
    bigram.load_from_file("./saves/model/bigram_prefix_save")
    trigram.load_from_file("./saves/model/trigram_prefix_save")
    five_gram.load_from_file("./saves/model/five_gram_prefix_save")

    N_grams = [
        (unigram, "unigram"),
        (bigram, "bigram"),
        (trigram, "trigram"),
        (five_gram, "five_gram"),
    ]

    dossier = "./archive/enwiki20201020"

    dataset = []

    for filename in os.listdir(dossier):
        # print(filename)
        if filename.endswith(".json") and random.randint(0,200) == 0:
            dataset.append(filename)
    
    res_perplexities = []

    for n_gram, name in N_grams:

        perplexities = []

        for filename in dataset:

            complete_path = os.path.join(dossier, filename)

            try:
                with open(complete_path, "r", encoding="utf-8") as f:
                    content = json.load(f)

                    for entry in content:
                        text = entry.get("text")

                        text = text.lower()
                        for punctuation in ";-*":
                            text = text.replace(punctuation, "")
                        for punctuation in ',.?":':
                            text = text.replace(punctuation, " " + punctuation + " ")

                        if text:  # Checks that "text" exists
                            perplexity = n_gram.perplexity(text)
                            perplexities.append(-perplexity)

            except Exception as e:
                continue
                # print(f"Error with {filename} : {e}")
        
        res_perplexities.append(statistics.mean(perplexities))

        print("perplexity {} :".format(name), res_perplexities[-1])
    
    
    plt.plot([1,2,3,5], res_perplexities, color='blue')

    plt.xticks([1, 2, 3, 5])

    plt.title("Perplexities")
    plt.xlabel("N (for the N-gram)")
    plt.ylabel("Log Perplexity")
    plt.legend()
    
    plt.savefig("perplexities.png")
    plt.show()

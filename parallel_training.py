from prefix_tree import N_gram
import statistics
import os
import json
import random
from english_dictionary.scripts.read_pickle import get_dict
import time
import ollama


def train_k_epochs_multiple(filename_save_text, model, k, n_grams):
    """Trains all the models k times, saves it to filename_save_model and saves the training data
    to filename_save_text. "model" is the name of the LLM used to generate the training data
    """
    english_dict = get_dict()

    time0 = time.time()

    for _ in range(k):

        time1 = time.time()

        key = random.choice(list(english_dict.keys()))
        del english_dict[key]
        code_prompt = f"Write something unique, interesting or intriguing about this word : {key}. Use a natural tone and avoid repeats"
        response = ollama.generate(model=model, prompt=code_prompt)

        time2 = time.time()

        for n_gram, filename_save_model in n_grams:

            text = ""
            for _ in range(n_gram.N - 1):
                text = "<s> " + text
            text = text + response["response"]
            text = text.lower()
            for punctuation in ";-*":
                text = text.replace(punctuation, "")
            for punctuation in ',.?":':
                text = text.replace(punctuation, " " + punctuation + " ")

            text = text + " </s>"

            with open(filename_save_text + filename_save_model, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n\n")

            filename_save_model = "saves/model/" + filename_save_model
    
            n_gram.load_from_file(filename_save_model)
            n_gram.train(text)

    for n_gram, filename_save_model in n_grams:
        filename_save_model = "saves/model/" + filename_save_model
        n_gram.save_to_file(filename_save_model)

    time3 = time.time()

    print("time to load the n-gram:", time1 - time0)
    print("time to generate:", time2 - time1)
    print("time to take care of the data and write:", time3 - time2)


if __name__ == "__main__":

    unigram = N_gram(1)
    bigram = N_gram(2)
    trigram = N_gram(3)
    five_gram = N_gram(5)
    ten_gram = N_gram(10)

    unigram.load_from_file("saves/model/unigram_prefix_save")
    bigram.load_from_file("saves/model/bigram_prefix_save")
    trigram.load_from_file("saves/model/trigram_prefix_save")
    five_gram.load_from_file("saves/model/five_gram_prefix_save")
    ten_gram.load_from_file("saves/model/ten_gram_prefix_save")

    N_grams = [
        (unigram, "unigram_prefix_save"),
        (bigram, "bigram_prefix_save"),
        (trigram, "trigram_prefix_save"),
        (five_gram, "five_gram_prefix_save"),
        (ten_gram, "ten_gram_prefix_save"),
    ]

    """ TO TRAIN THE MODELS """
    # while True:
    train_k_epochs_multiple(
        "saves/training_text/parallel_training_text_",
        "llama3.2:1b",
        1,
        N_grams,
    )
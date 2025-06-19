from prefix_tree import N_gram

if __name__ == "__main__":

    bigram = N_gram(2)

    """ TO TRAIN THE MODEL """
    while True:
        bigram.train_k_epochs(
            "saves/model/bigram_prefix_save",
            "saves/training_text/bigram_prefix_training_text",
            "llama3.2:1b",
            1,
        )

    """ TO GENERATE WITH THE MODEL """
    # bigram.load_from_file("saves/model/bigram_prefix_save")
    # bigram.generate(100)

    # print("")
    # print("Unigram Perplexity long :", unigram.perplexity("serenity is about cultivating a state of mind that is characterized by a sense of tranquility"))
    # print("Unigram Perplexity short :", unigram.perplexity("a sense of tranquility"))

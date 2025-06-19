from prefix_tree import N_gram

if __name__ == "__main__":

    unigram = N_gram(1)

    """ TO TRAIN THE MODEL """
    while True:
        unigram.train_k_epochs(
            "saves/model/unigram_prefix_save",
            "saves/training_text/unigram_prefix_training_text",
            "llama3.2:1b",
            1,
        )

    """ TO GENERATE WITH THE MODEL """
    # unigram.load_from_file("saves/model/unigram_prefix_save")
    # unigram.generate(100)

    # print("")
    # print("Unigram Perplexity long :", unigram.perplexity("serenity is about cultivating a state of mind that is characterized by a sense of tranquility"))
    # print("Unigram Perplexity short :", unigram.perplexity("a sense of tranquility"))

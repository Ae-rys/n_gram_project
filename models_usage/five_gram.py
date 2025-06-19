from prefix_tree import N_gram

if __name__ == "__main__":

    five_gram = N_gram(5)

    """ TO TRAIN THE MODEL """
    # while True:
    #     five_gram.train_k_epochs(
    #         "saves/model/five_gram_prefix_save",
    #         "saves/training_text/five_gram_prefix_training_text",
    #         "llama3.2:1b",
    #         1,
    #     )

    """ TO GENERATE WITH THE MODEL """
    five_gram.load_from_file("saves/model/five_gram_prefix_save")
    five_gram.generate(100)

    print("")
    print("Five_gram Perplexity long :", five_gram.perplexity("serenity is about cultivating a state of mind that is characterized by a sense of tranquility"))
    # print("Five_gram Perplexity short :", five_gram.perplexity("the cat sat on the mat "))
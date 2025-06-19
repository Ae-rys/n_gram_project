from prefix_tree import N_gram

if __name__ == "__main__":

    ten_gram = N_gram(10)

    """ TO TRAIN THE MODEL """
    while True:
        ten_gram.train_k_epochs(
            "saves/model/ten_gram_prefix_save",
            "saves/training_text/ten_gram_prefix_training_text",
            "llama3.2:1b",
            1,
        )

    """ TO GENERATE WITH THE MODEL """
    # ten_gram.load_from_file("saves/model/ten_gram_prefix_save")
    # ten_gram.generate(100)

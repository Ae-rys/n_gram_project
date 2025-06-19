import json
from english_dictionary.scripts.read_pickle import get_dict
import time
import os
import random
import ollama
import math

"""Classes for a N_gram. I use entire words as tokens. """

VAL_NEXT = 10

NUMBER_NGRAM = "N"
DICT_OCC = "D"
CHILDREN = "C"


class TrieNode:
    def __init__(self):
        self.children = {}
        self.dict_occ = {}
        self.number_ngram = 0

    def to_dict(self):
        """To save the trie in a json"""
        return {
            NUMBER_NGRAM: self.number_ngram,
            DICT_OCC: self.dict_occ,
            CHILDREN: {
                word: child.to_dict() for word, child in self.children.items()
            },
        }

    @staticmethod
    def from_dict(data):
        """to download the trie from a json"""
        node = TrieNode()
        node.number_ngram = data[NUMBER_NGRAM]
        node.dict_occ = data.get(DICT_OCC, {})
        for word, child_data in data[CHILDREN].items():
            node.children[word] = TrieNode.from_dict(child_data)
        return node


class PrefixTree:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, n_gram):
        """adds an n_gram to the trie"""
        node = self.root
        last_word = n_gram[-1]

        node.number_ngram += 1
        if last_word in node.dict_occ:
            node.dict_occ[last_word] += 1
        else:
            node.dict_occ[last_word] = 1

        n_gram_rev = reversed(n_gram[:-1])

        for word in n_gram_rev:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]

            node.number_ngram += 1
            if last_word in node.dict_occ:
                node.dict_occ[last_word] += 1
            else:
                node.dict_occ[last_word] = 1

    def search(self, n_gram):
        """returns the number of occurences of a n_gram in the trie."""

        last_word = n_gram[-1]

        n_gram_cut = n_gram[:-1]

        node = self._find_node(n_gram_cut)
        if node == None or last_word not in node.dict_occ:
            return 0
        return node.dict_occ[last_word]

    def _find_node(self, tokens, N):
        """returns the node associated with tokens. Reversed search"""
        node = self.root

        profondeur = 1

        # We choose the context we want to generate the next word (based on N and VAL_NEXT)
        while profondeur < N and profondeur <= len(tokens):
            if (
                tokens[-profondeur] in node.children
                and node.children[tokens[-profondeur]].number_ngram >= VAL_NEXT
            ):
                node = node.children[tokens[-profondeur]]
                profondeur += 1
            else:
                break

        if node == None:
            assert False  # ce n'est pas censé arriver

        return node

    def save_to_file(self, filename):
        """Saves the trie in filename"""
        trie_dict = self.root.to_dict()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(trie_dict, f, ensure_ascii=False)

    def load_from_file(self, filename):
        """Downloads the trie from filename"""
        if os.stat(filename).st_size == 0:
            self.root = TrieNode()
        else:
            with open(filename, "r", encoding="utf-8") as f:
                trie_dict = json.load(f)
                self.root = TrieNode.from_dict(trie_dict)


class N_gram:
    def __init__(self, N):
        self.trie = PrefixTree()
        self.N = N

    def add(self, n_gram):
        """Adds a n_gram to self.trie."""
        self.trie.insert(n_gram)

    def count(self, n_gram):
        """Returns the number of apparitions of a n_gram."""
        return self.trie.search(n_gram)

    def probability(self, tokens):
        """Returns the probability of the last word of an n_gram given the n last words"""
        n_gram = tokens[-self.N : -1]

        last_word = tokens[-1]

        node = self.trie.root

        profondeur = 1

        # We choose the context we want to generate the next word (based on N and VAL_NEXT)
        while profondeur < self.N and profondeur <= len(n_gram):
            if (
                n_gram[-profondeur] in node.children
                and node.children[n_gram[-profondeur]].number_ngram >= VAL_NEXT
                and last_word in node.children[n_gram[-profondeur]].dict_occ.keys()
            ):
                node = node.children[n_gram[-profondeur]]
                profondeur += 1
            else:
                break

        if node == None:
            assert False  # ce n'est pas censé arriver

        total = node.number_ngram

        if total == 0:
            return 0.0
        if last_word in node.dict_occ:
            return node.dict_occ[last_word] / total
        else:
            # print(last_word, "has never been seen")
            return 1 / total  # If the word has never been seen, we return something

    def choose_next_word(self, tokens):
        """Returns a next word given a list of tokens"""

        node = self.trie._find_node(tokens, self.N)

        keys = list(node.dict_occ.keys())
        values = list(node.dict_occ.values())

        random_key = random.choices(keys, weights=values)

        return random_key[0]

    def train(self, text):
        """Trains the model on the words in text"""

        text = text.split()

        for i in range(len(text) - self.N + 1):
            self.add(text[i : i + self.N])

    # TO MODIFY
    def generate(self, nbr_words):
        """Generates using the model, starting with '<s>', the character marking the beginning of a prompt"""
        text = "<s> "
        print("")

        for i in range(nbr_words):
            next_word = self.choose_next_word(text.split())
            text += " " + next_word
            if next_word == "</s>":
                break
            else:
                print(next_word, end=" ")

        print("")

    def perplexity(self, text):
        text = text.split()

        perplexity = 0

        for i in range(1, self.N):
            n_gram = text[:i]
            perplexity += math.log(self.probability(n_gram))

        for i in range(len(text) - self.N + 1):
            n_gram = text[i : i + self.N]
            perplexity += math.log(self.probability(n_gram))

        return perplexity * (1 / len(text))

    def save_to_file(self, filename):
        """Saves the model to filename"""
        self.trie.save_to_file(filename)

    def load_from_file(self, filename):
        """loads the model from filename"""
        self.trie.load_from_file(filename)

    def train_k_epochs(self, filename_save_model, filename_save_text, model, k):
        """Trains the model k times, saves it to filename_save_model and saves the training data
        to filename_save_text. "model" is the name of the LLM used to generate the training data
        """
        english_dict = get_dict()

        time0 = time.time()

        self.load_from_file(filename_save_model)

        for _ in range(k):

            time1 = time.time()

            key = random.choice(list(english_dict.keys()))
            del english_dict[key]
            code_prompt = f"Write something unique, interesting or intriguing about this word : {key}. Use a natural tone and avoid repeats"
            response = ollama.generate(model=model, prompt=code_prompt)

            time2 = time.time()

            text = ""
            for _ in range(self.N - 1):
                text = "<s> " + text
            text = text + response["response"]
            text = text.lower()
            for punctuation in ";-*":
                text = text.replace(punctuation, "")
            for punctuation in ',.?":':
                text = text.replace(punctuation, " " + punctuation + " ")

            text = text + " </s>"

            with open(filename_save_text, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n\n")

            self.train(text)

        self.save_to_file(filename_save_model)

        time3 = time.time()

        print("time to load the n-gram:", time1 - time0)
        print("time to generate:", time2 - time1)
        print("time to take care of the data and write:", time3 - time2)

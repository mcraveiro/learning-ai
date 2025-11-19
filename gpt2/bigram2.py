import json
import random
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        """Train on a list of tokens."""
        tokens = ["<s>"] + text + ["</s>"]  # Add sentence boundary tokens
        print("tokens: ", tokens)

        for i in range(1, len(tokens)):
            print("----- i: ", i)
            w_prev = tokens[i-1]
            print("w_prev: ", w_prev)
            w_curr = tokens[i]
            print("w_curr: ", w_curr)

            self.unigram_counts[w_prev] += 1
            self.bigram_counts[w_prev][w_curr] += 1

            self.vocab.update([w_prev, w_curr])

        print("unigram counts: ", self.unigram_counts)
        print("bigram_counts: ", json.dumps(self.bigram_counts, indent=2))

    # In summary, the function calculates the probability that a given word
    # (w_curr) follows another given word (w_prev) in the text, without any
    # smoothing techniques applied. This is useful for understanding the
    # likelihood of word sequences in a language model but can be problematic
    # if some words are very rare or absent from the dataset.
    def bigram_prob(self, w_prev, w_curr):
        """Return probability P(curr | prev). Un-smoothed."""
        prev_count = self.unigram_counts[w_prev]
        if prev_count == 0:
            return 0
        return self.bigram_counts[w_prev][w_curr] / prev_count

    def generate(self, max_len=20):
        """Generate text from the model."""
        word = "<s>"
        output = []

        for _ in range(max_len):
            next_words = list(self.bigram_counts[word].keys())
            if not next_words:
                break
            # In summary, this code snippet is designed to calculate the
            # conditional probabilities of a sequence of words given a specific
            # preceding word. It is useful for tasks such as language modeling,
            # where you need to understand the likelihood of different words
            # appearing next in a sequence. The list =probs= will contain these
            # probabilities for each word in =next_words=.
            probs = [
                self.bigram_prob(word, w)
                for w in next_words
            ]
            word = random.choices(next_words, weights=probs, k=1)[0]
            if word == "</s>":
                break
            output.append(word)

        return " ".join(output)


# --------------------------------------------
# Example usage:
# --------------------------------------------

random.seed(1337)

corpus = "the cat sat on the mat the cat ate food".split()

model = BigramModel()
model.train(corpus)

print("P(cat | the) =", model.bigram_prob("the", "cat"))
print("Generated text:", model.generate())

# with open('input/input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# model = BigramModel()
# model.train(text.split())


# print("P(Henry | King) =", model.bigram_prob("King", "Henry"))
# print("Generated text:", model.generate())

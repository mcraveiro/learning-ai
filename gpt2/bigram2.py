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

        for i in range(1, len(tokens)):
            w_prev = tokens[i-1]
            w_curr = tokens[i]

            self.unigram_counts[w_prev] += 1
            self.bigram_counts[w_prev][w_curr] += 1

            self.vocab.update([w_prev, w_curr])

    def bigram_prob(self, w_prev, w_curr):
        """Return probability P(curr | prev). Unsmoothened."""
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

#  random.seed(1337)

corpus = "the cat sat on the mat the cat ate food".split()

model = BigramModel()
model.train(corpus)

print("P(cat | the) =", model.bigram_prob("the", "cat"))
print("Generated text:", model.generate())

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

model = BigramModel()
model.train(text.split())


print("P(Henry | King) =", model.bigram_prob("King", "Henry"))
print("Generated text:", model.generate())

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

with open('input/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


def main():


    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    print(encode("hii there"))
    print(decode(encode("hii there")))

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    print(data.shape, data.dtype)
    print(data[:1000])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context.tolist()} the target: {target}")

if __name__ == "__main__":
    main()

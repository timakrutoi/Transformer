import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet = {',': 0, '(': 1, 'q': 2, 'c': 3, 'g': 4, '\\': 5, '7': 6, 'C': 7, 'n': 8, 'e': 9, '¬': 10, '9': 11, 'x': 12, '.': 13, '5': 14, 't': 15, 'h': 16, 'P': 17, "'": 18, 'i': 19, 'M': 20, 'A': 21, '?': 22, 'N': 23, 'D': 24, 'j': 25, ';': 26, '0': 27, '4': 28, 'G': 29, 'O': 30, 'a': 31, 'u': 32, 'w': 33, 'X': 34, ']': 35, 'R': 36, 'W': 37, '3': 38, 'y': 39, 'Y': 40, '6': 41, '\n': 42, '*': 43, 'z': 44, '!': 45, 'p': 46, 'K': 47, 'F': 48, ':': 49, '”': 50, '/': 51, '‘': 52, 'L': 53, 'T': 54, '"': 55, '2': 56, '—': 57, '§': 58, 'V': 59, 'r': 60, 'J': 61, '“': 62, 'f': 63, 'k': 64, 'm': 65, 'l': 66, 'b': 67, '>': 68, 'E': 69, 'S': 70, '^': 71, 'I': 72, '8': 73, 'H': 74, '1': 75, 'o': 76, 'd': 77, '-': 78, 'v': 79, 'U': 80, ' ': 81, ')': 82, 's': 83, 'Z': 84, 'Q': 85, '’': 86, 'B': 87}
        self.stoi = {s:int(i) for i, s in enumerate(alphabet)}
        self.itos = {int(i):s for i, s in enumerate(alphabet)}

        self.size = len(list(self.stoi.keys()))
        # print(self.stoi)

    def encode(self, x):
        res = torch.empty(len(x))
        for i, l in enumerate(x):
            res[i] = self.stoi[l]
        return res

    def decode(self, x):
        res = []
        for i, l in enumerate(x):
            res.append(self.itos[int(l.item())])
        return ''.join(res)

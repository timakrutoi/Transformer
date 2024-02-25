import regex as re
import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet = {',': 0, '(': 1, 'q': 2, 'c': 3, 'g': 4, '\\': 5, '7': 6, 'C': 7, 'n': 8, 'e': 9, '¬': 10, '9': 11, 'x': 12, '.': 13, '5': 14, 't': 15, 'h': 16, 'P': 17, "'": 18, 'i': 19, 'M': 20, 'A': 21, '?': 22, 'N': 23, 'D': 24, 'j': 25, ';': 26, '0': 27, '4': 28, 'G': 29, 'O': 30, 'a': 31, 'u': 32, 'w': 33, 'X': 34, ']': 35, 'R': 36, 'W': 37, '3': 38, 'y': 39, 'Y': 40, '6': 41, '\n': 42, '*': 43, 'z': 44, '!': 45, 'p': 46, 'K': 47, 'F': 48, ':': 49, '”': 50, '/': 51, '‘': 52, 'L': 53, 'T': 54, '"': 55, '2': 56, '—': 57, '§': 58, 'V': 59, 'r': 60, 'J': 61, '“': 62, 'f': 63, 'k': 64, 'm': 65, 'l': 66, 'b': 67, '>': 68, 'E': 69, 'S': 70, '^': 71, 'I': 72, '8': 73, 'H': 74, '1': 75, 'o': 76, 'd': 77, '-': 78, 'v': 79, 'U': 80, ' ': 81, ')': 82, 's': 83, 'Z': 84, 'Q': 85, '’': 86, 'B': 87}  # noqa
        self.stoi = {s: int(i) for i, s in enumerate(alphabet)}
        self.itos = {int(i): s for i, s in enumerate(alphabet)}

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


class BPETokenizer(nn.Module):
    def __init__(self):

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, data, n_merges, save=True):
        ids = list(self.encode(data))
        sl, sv = len(ids), len(self.vocab)

        for _ in range(n_merges):
            stats = {}
            for a, b in zip(ids, ids[1:]):
                try:
                    stats[(a, b)] += 1
                except KeyError:
                    stats[(a, b)] = 1

            mc = max(stats, key=stats.get)
            self.merges[mc] = len(self.vocab)
            self.vocab[len(self.vocab)] = self.vocab[mc[0]] + self.vocab[mc[1]]

            ids = self._merge_pair(ids, mc, self.merges[mc])

        print('Training complete:')
        print('Num merges:', n_merges)
        print(f'New vocab size: {len(self.vocab)}',
              f'({len(self.vocab)/sv:.02f}x increase)')
        print(f'Compression factor {sl/len(ids):.02f}')

        if save:
            self.save()

    def save(self):
        torch.save({'vocab': self.vocab, 'merge': self.merges}, 'token-info')

    def load(self):
        c = torch.load('token-info')
        self.vocab = c['vocab']
        self.merges = c['merge']

    def __str__(self):
        return f'Tokenizer(Vocab size: {len(self.vocab)}, \
Num merges: {len(self.merges)})'

    def _merge_pair(self, ids, mc, idx):
        tmp = []
        i = 0
        while i < len(ids):
            if len(ids) - i > 2 and (ids[i], ids[i+1]) == mc:
                tmp.append(idx)
                i += 2
            else:
                tmp.append(ids[i])
                i += 1
        return list(tmp)

    def encode(self, text):
        ids = list(text.encode('utf-8'))

        for p, k in self.merges.items():
            ids = self._merge_pair(ids, p, k)

        return ids

    def decode(self, ids, debug=False):
        if debug:
            text = [x.decode('utf-8', errors='replace') for x in list(self.vocab[i] for i in ids)]
            return text

        text = list(b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace'))
        return ''.join(text)


class GPT2Tokenizer(BPETokenizer):
    def __init__(self, filter_=None):
        super().__init__()
        if filter_ is None:
            filter_ = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.filter = re.compile(filter_)

    def encode(self, text):
        ids = [super(GPT2Tokenizer, self).encode(x) for x in re.findall(self.filter, text)]
        return sum(ids, [])


if __name__ == '__main__':
    from argparse import ArgumentParser as AP

    p = AP()

    p.add_argument('-t', '--train-num-merges', type=int,
                   help='Total number of merges (retrains tokenizer).')
    p.add_argument('-s', '--text', type=str,
                   help='Text to endcode and decode.')
    p.add_argument('--train-data', type=str, default='text.txt',
                   help='Text to train tokenizer on (if training).')
    p.add_argument('--tokenizer', type=str, choices=['simple', 'bpe', 'gpt2'], default='gpt2',
                   help='Tokenizer to use.')

    args = p.parse_args()

    tks = {
        'simple': Tokenizer,
        'bpe': BPETokenizer,
        'gpt2': GPT2Tokenizer,
    }

    t = tks[args.tokenizer]()

    if args.train_num_merges:
        with open(args.train_data, 'r') as f:
            t.train(f.read(), args.train_num_merges)
    else:
        t.load()

    if args.text is None:
        args.text = 'the quick brown fox jumps over the lazy dog'

    print(t.encode(args.text))
    print(t.decode(t.encode(args.text), True))

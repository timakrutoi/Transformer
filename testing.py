import tiktoken
import torch
import torch.nn.functional as F

from model import Transformer
from tokenizer import Tokenizer


def generate(model, tokenizer, start_token, n_tokens, block_size, vocab_size):
    sec = tokenizer.encode(start_token).unsqueeze(0)
    for t in range(n_tokens):
        # print(sec[:, -block_size:].shape, block_size)
        logits = model(sec[:, -block_size:].long())[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next = torch.multinomial(probs, 1)

        sec = torch.cat((sec, next), dim=1)
    print(tokenizer.decode(sec[0]))
    

if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('-p', '--prompt', type=str, default='\n')
    p.add_argument('-n', '--num-tokens', type=int, default=100)
    p.add_argument('-m', '--model-path', type=str, required=True)

    a = p.parse_args()

    cp = torch.load(a.model_path)
    params = cp['params']

    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab

    model = Transformer(vocab_size, params.block_size, params.num_emb, params.heads_num, params.block_num)
    model.load_state_dict(cp['model_state_dict'])

    generate(model, tokenizer, a.prompt, a.num_tokens, params.block_size, vocab_size)

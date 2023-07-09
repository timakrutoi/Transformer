import numpy as np
import tiktoken
import torch
from torch.optim import Adam
from tqdm import tqdm

from model import Transformer
from tokenizer import Tokenizer
from dataloader import get_dataloader


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('--bl', '--block-size', dest='block_size', type=int, default=128)
    p.add_argument('--bs', '--batch-size', dest='batch_size', type=int, default=10)
    p.add_argument('--hn', '--heads-num', dest='heads_num', type=int, default=4)
    p.add_argument('--bn', '--block-num', dest='block_num', type=int, default=4)
    p.add_argument('--ne', '--num-emb', dest='num_emb', type=int, default=80)
    p.add_argument('-e', '--num-epochs', type=int, default=100)
    p.add_argument('-d', '--data-path', type=str, default='tolkien.txt')
    p.add_argument('--lr', type=float, default=1e-5)

    a = p.parse_args()
    for k, v in vars(a).items():
        print(f'{k}: {v}')

    save_cp = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab
    print(f'tokenizer: {vocab_size}')

    model = Transformer(vocab_size, a.block_size, a.num_emb, a.heads_num, a.block_num).to(device)
    optimizer = Adam(Transformer.parameters(model), lr=a.lr)

    crit = torch.nn.CrossEntropyLoss()

    with open(a.data_path, 'r') as f:
        data = tokenizer.encode(f.read())
        data = torch.tensor(data)

    train, test = get_dataloader(data, a.batch_size, a.block_size)

    print(f'train: {len(train)}')
    print(f'test: {len(test)}')

    for e in range(a.num_epochs):
        it = tqdm(train, desc=f'training Epoch={e}')
        for x, t in it:
            optimizer.zero_grad()
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            loss = crit(y.view(-1, vocab_size), t.view(-1))
            loss.backward()
            optimizer.step()

            it.set_postfix(str=f'{loss.item()=}')

        it.close()

        if not e % 1:
            with torch.no_grad():
                losses = []

                it = tqdm(test, desc=f'testing Epoch={e}')
                for x, t in it:
                    x = x.to(device)
                    t = t.to(device)
                    y = model(x)
                    loss = crit(y.view(-1, vocab_size), t.view(-1))
                    losses.append(loss)
                    it.set_postfix(str=f'{np.mean(losses).item()=}')

                it.close()

            if save_cp:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'loss': loss,
                    'params': a,
                }, 'model_e{}_loss{:.3f}'.format(e, round(loss.item(), 2)))

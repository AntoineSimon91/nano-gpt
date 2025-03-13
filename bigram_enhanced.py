import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independant sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

class Tokeniser:
    def __init__(self, text):
        self.text=text
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        self.create_mappings()

    def create_mappings(self):
        #string to integers
        self.stoi = {ch: i for i, ch in enumerate(self.characters)}
        #intergers to strings
        self.itos = {i: ch for i, ch in enumerate(self.characters)}

    def encode(self, string):
        """Convert a string into a list of integers."""
        return [self.stoi[c] for c in string]

    def decode(self, encoded_list):
        """Convert a list of integers back into a string."""
        return ''.join([self.itos[i] for i in encoded_list])

class Dataset:
    def __init__(self, tokeniser, data, split_ratio=0.9):
        self.data = torch.tensor(tokeniser.encode(data), dtype=torch.long)
        self.split_ratio = split_ratio
        self.split()

    def split(self):
        n = int(0.9*len(self.data))
        self.train = self.data[:n]
        self.validation = self.data[n:]

    def get_batch(self, split='train'):
        """generate a small batch of data of inputs x and targets y"""
        data = self.train if split == 'train' else self.validation
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x,y

@torch.no_grad()
def estimate_loss(dataset):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y= dataset.get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # Make sure that features do not communicate with the past
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out=  torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads # each head has a size of n_embd / num_heads
        self.multi_head_attention = MultiHeadAttention(num_heads, head_size) # multi-head attention
        self.feed_forward = FeedForward(n_embd) # feed forward network
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x)) # apply multi-head attention
        x = x + self.feed_forward(self.ln2(x)) # apply feed forward network

        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # each token gets its own embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position gets its own embedding
        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # maps the embedding to the vocabulary size


    def forward(self, idx, targets=None):

        B, T = idx.shape # B is batch size, T is sequence length
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # apply the blocks
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == "__main__":
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()


    tokeniser = Tokeniser(text)
    encoded = tokeniser.encode("Hello, world!")
    print("Encoded:", encoded)
    decoded = tokeniser.decode(encoded)
    print("Decoded:", decoded)

    dataset = Dataset(tokeniser, text)
    print(dataset.get_batch())
    model = BigramLanguageModel(tokeniser.vocab_size)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(dataset)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = dataset.get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokeniser.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
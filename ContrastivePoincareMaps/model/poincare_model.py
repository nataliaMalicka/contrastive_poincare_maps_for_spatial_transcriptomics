import torch
import torch.nn as nn
import torch.nn.functional as F
from model.poincare_manifold import PoincareBall

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, out_dim =None, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        if out_dim:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        else:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)

class Hypblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, out_dim=None, dropout=0, bias=True):

        super().__init__()

        self.manifold = PoincareBall()
        self.c = 1.0
        self.dropout = dropout
        self.use_bias = bias
        in_dim = input_dim
        self.weights = nn.ParameterDict({})
        self.biases = nn.ParameterDict({})
        for i in range(n_layers):
            self.weights["layer_"+str(i)] = nn.Parameter(torch.Tensor(hidden_dim, in_dim), requires_grad=True)
            self.biases["layer_"+str(i)] = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
            in_dim = hidden_dim
        self.weights["layer_"+str(n_layers)] = nn.Parameter(torch.Tensor(out_dim, in_dim), requires_grad=True)
        self.biases["layer_"+str(n_layers)] = nn.Parameter(torch.Tensor(out_dim), requires_grad=True)
        self.n_layers = n_layers
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights.keys():
            torch.nn.init.xavier_uniform_(self.weights[w])
        for b in self.biases.keys():
            self.biases[b].data.fill_(0.01)

    def forward(self, x):
        x_hyp = self.manifold.proj(self.manifold.expmap0(x, self.c), self.c)
        h = x_hyp
        for i in range(self.n_layers+1):
            w = self.weights["layer_"+str(i)]
            b = self.biases["layer_"+str(i)]
            w = F.dropout(w, self.dropout, training=self.training)
            h = self.manifold.proj(self.manifold.mobius_matvec(w, h, c=self.c), self.c)
            if self.use_bias:
                bias = self.manifold.proj_tan0(b.view(1, -1), self.c)
                hyp_bias = self.manifold.expmap0(bias, self.c)
                hyp_bias = self.manifold.proj(hyp_bias, self.c)
                h = self.manifold.mobius_add(h, hyp_bias, c=self.c)
                h = self.manifold.proj(h, self.c)
            if i != self.n_layers:
                h = self.act(self.manifold.proj_tan0(self.manifold.logmap0(h, self.c), self.c))
                h = self.manifold.proj(self.manifold.expmap0(h, self.c), self.c)
        return h

class CPM(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_depth=1,
        hyp_depth = 1,
        corruption_rate=0.6
    ):

        super().__init__()

        self.hyp_flag = 0
        self.enc_flag = 0

        if hyp_depth:
            self.hyp_flag = 1
        if encoder_depth:
            self.enc_flag = 1

        if self.enc_flag:
            self.encoder = MLP(input_dim, emb_dim, encoder_depth)
            self.encoder.apply(self._init_weights)
            self.Hyp = Hypblock(emb_dim, emb_dim, hyp_depth, out_dim=2)
        else:
            self.Hyp = Hypblock(input_dim, emb_dim, hyp_depth, out_dim=2)

        self.corruption_len = int(corruption_rate * input_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_sample):
        batch_size, m = anchor.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        # modified by me: removing CL
        """
        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool, device=anchor.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = torch.where(corruption_mask, random_sample, anchor)
        """

        # compute embeddings
        if self.enc_flag:
            emb_anchor = self.encoder(anchor)
            # emb_positive = self.encoder(positive) # MYZ
            emb_anchor = self.Hyp(emb_anchor)
            # emb_positive = self.Hyp(emb_positive) # MYZ
        else:
            emb_anchor = self.Hyp(anchor)
            # emb_positive = self.Hyp(positive) # MYZ

        # modified by me: removing CL
        # return emb_anchor, emb_positive
        return emb_anchor, None  # emb_positive MYZ

    def get_embeddings(self, input):
        if self.enc_flag:
            emb = self.encoder(input)
            emb = self.Hyp(emb)
        else:
            emb = self.Hyp(input)
        return emb

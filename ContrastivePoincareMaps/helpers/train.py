import torch
from torch.utils.data import DataLoader
from model.poincare_model import CPM
from optimiser.radam import RiemannianAdam
from model.poincare_loss import Poincare_loss
from tqdm.auto import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import random
import numpy as np
from sklearn.metrics import pairwise_distances

#from ContrastivePoincareMaps.model.poincare_loss import WeightedKNNLoss
from model.poincare_loss import WeightedKNNLoss
from sklearn.metrics.pairwise import pairwise_distances as pdist
from sklearn.neighbors import kneighbors_graph

#from ContrastivePoincareMaps.model.poincare_loss import WeightedKNNLoss


def fix_seed(seed):
    random.seed(seed)               # for py
    np.random.seed(seed)            # for np
    torch.manual_seed(seed)         # for cpu
    torch.cuda.manual_seed(seed)    # for gpu

def train_epoch(model, criterion, train_loader, opt1, opt2, device, epoch):
    """
    opt1: euclidean encoder (mlp), opt2: hyperbolic, different lr dep on eucl vs hyperbolic (smaller lr)
    train_loader: pt dataloader that loads data into batches
    """
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for anchor, positive, indices in batch: # MYZ
    #for anchor, positive in batch:
        anchor, positive = anchor.to(device), positive.to(device)

        # reset gradients
        if opt1:
            opt1.zero_grad()
        if opt2:
            opt2.zero_grad()


        # get embeddings
        # emb_anchor, emb_positive = model(anchor, positive)
        # MYZ
        emb_anchor, _ = model(anchor, positive) # MYZ


        # compute loss
        # loss = criterion(emb_anchor, emb_positive)
        # MYZ
        loss = criterion(emb_anchor, indices)  # emb_positive) # MYZ
        loss.backward()

        # update model weights
        if opt1:
            opt1.step()
        if opt2:
            opt2.step()

        # log progress
        epoch_loss += anchor.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)

def train_model(args, data, seed =0):

    fix_seed(seed)

    #Model hyperparameters
    batch_size = args.batch_size
    #batch_size = len(data) # full dataset as one batch MYZ
    epochs = args.n_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dim = args.emb_dim
    earlystop = args.early_stop

    #lr = args.lr


    #/////////////////////////////////////////////////////////////////
    D_high = pdist(data.proc_data, metric='euclidean')

    k = 50

    # distances to k neighbors
    A = kneighbors_graph(
        data.proc_data,
        k,
        mode='distance',
        include_self=False
    )

    A = A.toarray()

    # compute sigma - dist scale
    sigma = np.mean(A[A > 0])

    # convert distances to Gauss w
    W = np.exp(-(A ** 2) / (2 * sigma ** 2))

    # zero out non-neighbors
    W[A == 0] = 0.0

    #loss_fn = WeightedKNNLoss(W, lambda_rep=1.5)
    loss_fn = WeightedKNNLoss(torch.from_numpy(W).float(), lambda_rep=1.5)
    #/////////////////////////////////////////////////////////////////////


    # initialise dataloader
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # # CHANGED FOR DATASET MOUSE GESTRICULATION
    # train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    #initialise model
    model = CPM(
        input_dim=data.shape[1],
        emb_dim=emb_dim,
        encoder_depth= args.n_enc_layers,
        hyp_depth= args.n_hyp_layers,
        corruption_rate=args.corr_rate
    ).to(device)

    #initialise optimisers - optimisers with different learning rates for projection and hyperbolic layers
    if args.n_enc_layers > 0:
        opt1 = RiemannianAdam(model.encoder.parameters(), lr=args.lr_proj)
    else:
        opt1 =None
    opt2 = RiemannianAdam(model.Hyp.parameters(), lr=args.lr_hyp)

    # training and logs
    #loss_fn = Poincare_loss()
    # D_high = pdist(data.proc_data, metric='euclidean')


    loss_history = []
    earlystop_count = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, loss_fn, train_loader, opt1, opt2, device, epoch)
        loss_history.append(epoch_loss)
        if epoch > 10:
            delta = abs(loss_history[epoch - 1] - loss_history[epoch - 2])
            if (delta < earlystop):
                earlystop_count += 1
            if earlystop_count > 50:
                print(f'\nStopped at epoch {epoch}')
                break
        if epoch % 50 == 0:
            int_emb = dataset_embeddings(model, train_loader, device)
            ball_norm = np.sqrt(int_emb[:, 0] ** 2 + int_emb[:, 1] ** 2)
            if np.max(ball_norm) > 1.001:
                print('The learning rate is too high.')

    #save plots, model and embedding
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(loss_history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.savefig(args.res_dir + 'losscurve.png')

    torch.save(model, args.res_dir + "cpm_model.pt")

    train_data = DataLoader(data, batch_size=batch_size, shuffle=False)
    embeddings = dataset_embeddings(model, train_data, device)
    emb = pd.DataFrame(embeddings)
    emb.to_csv(args.res_dir + args.fname + "_emb.csv", index=False, header=False)

    return embeddings

def dataset_embeddings(model, loader, device):
    model.eval().to(device)
    embeddings = []

    with torch.no_grad():
        for anchor, _, _ in tqdm(loader): # MYZ
        #for anchor, _ in tqdm(loader):
            anchor = anchor.to(device)
            embeddings.append(model.get_embeddings(anchor))

    embeddings = torch.cat(embeddings).cpu().numpy()

    return embeddings

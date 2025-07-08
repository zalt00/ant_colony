import mylib
import importlib
mylib = importlib.reload(mylib)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch.nn import Sequential, Linear, ReLU

class EdgePolicy(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        # MLP pour GINEConv
        gin_nn1 = Sequential(Linear(node_feat_dim, hidden_dim),
                             ReLU(),
                             Linear(hidden_dim, hidden_dim))
        gin_nn2 = Sequential(Linear(hidden_dim, hidden_dim),
                             ReLU(),
                             Linear(hidden_dim, hidden_dim))

        self.conv1 = GINEConv(gin_nn1, edge_dim=2)
        self.conv2 = GINEConv(gin_nn2, edge_dim=2)

        # head pour scorer chaque arête
        self.edge_mlp = Sequential(
            Linear(2*hidden_dim + 2, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Message passing
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = self.conv2(h, edge_index, edge_attr)

        # 2. Préparer embeddings d'arêtes
        src, dst = edge_index
        h_edge = torch.cat([h[src], h[dst], edge_attr], dim=-1)

        # 3. Scores bruts
        logits = self.edge_mlp(h_edge).squeeze(-1)
        return logits


helper = mylib.Helper()

edges = edge_index = edge_types = edge_to_id = num_nodes = None
def update_graph():
    global edges, edge_index, edge_types, edge_to_id, num_nodes
    edges = helper.get_dataset()
    num_nodes = max(max(u,v) for u,v,t in edges) + 1

    # Séparer arrays pour plus de rapidité
    edge_index = torch.tensor([[u for u,v,t in edges],
                            [v for u,v,t in edges]], dtype=torch.long)
    edge_types = torch.tensor([t for u,v,t in edges], dtype=torch.long)

    edge_to_id = {frozenset((u, v)): i for (i, (u, v, _)) in enumerate(edges)}

update_graph()
# Récupération des arêtes
# Node features initiales (ici, vecteurs unité)
node_feat_dim = 1
x_init = torch.ones((num_nodes, node_feat_dim), dtype=torch.float)


from torch.distributions import Categorical

# Hyperparamètres
hidden_dim = 64
lr = 1e-3
episodes = 500
max_steps = 20 # len(edges)  # au pire on flippes toutes les arêtes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgePolicy(node_feat_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(episodes):
    # Réinitialiser l'état
    helper.reset(ep)
    update_graph()
    types = edge_types
    x = x_init.to(device)

    episode_loss = 0.0
    episode_reward = 0.0

    for step in range(max_steps):
        # Construire edge_attr sous forme one-hot
        edge_attr = F.one_hot(types, num_classes=2).to(torch.float)

        # Forward
        logits = model(x, edge_index.to(device), edge_attr)

        # Masque arêtes de type 0
        mask0 = (types == 0).to(device)
        if mask0.sum() == 0:
            break  # plus d'arêtes à flipper

        logits0 = logits[mask0]
        probs0 = F.softmax(logits0, dim=0)
        dist = Categorical(probs0)

        # Échantillonnage de l’action
        a_idx = dist.sample()  # indice dans les arêtes type0
        logp = dist.log_prob(a_idx)

        # Traduire en index global
        global_idx = mask0.nonzero()[a_idx]

        # Exécuter l’action : flip
        (r, (u2, v2)) = helper.recompense(global_idx)  # récompense

        # Mise à jour temporaire de l’état
        types[global_idx] = 1
        idx2 = edge_to_id[frozenset((u2, v2))]
        types[idx2] = 0

        # Accumuler perte et récompense
        episode_loss += -logp * r
        episode_reward += r

    # Backprop et mise à jour des paramètres
    optimizer.zero_grad()
    episode_loss.backward()
    optimizer.step()

    if (ep+1) % 50 == 0:
        print(f"Épisode {ep+1:03d} | Réward cumulé {episode_reward:.2f}")



import random
rm = 0.0
rr = 0.0
rc = 0
for iter_id in range(50000):
    helper.reset(iter_id)
    update_graph()
    types = edge_types

    x = x_init.to(device)

    episode_loss = 0.0
    episode_reward = 0.0

    # Construire edge_attr sous forme one-hot
    edge_attr = F.one_hot(types, num_classes=2).to(torch.float)

    # Forward
    logits = model(x, edge_index.to(device), edge_attr)

    # Masque arêtes de type 0
    mask0 = (types == 0).to(device)
    if mask0.sum() == 0:
        break  # plus d'arêtes à flipper

    logits0 = logits[mask0]
    probs0 = F.softmax(logits0, dim=0)
    dist = Categorical(probs0)

    # Échantillonnage de l’action
    a_idx = dist.sample()  # indice dans les arêtes type0
    logp = dist.log_prob(a_idx)

    # Traduire en index global
    global_idx = mask0.nonzero()[a_idx]

    global_idx2 = mask0.nonzero()[random.randrange(0, mask0.sum())]


    # Exécuter l’action : flip
    (r, (u2, v2)) = helper.recompense(global_idx)  # récompense

    helper.reset(0)
    (r2, (u2, v2)) = helper.recompense(global_idx2)  # récompense

    rm += r
    rr += r2
    rc += 1
    if iter_id % 1000 == 0.0:
        print("{} {} {}", iter_id, rm/rc, rr/rc)

print(rm / rc, rr / rc)




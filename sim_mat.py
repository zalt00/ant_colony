import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from tqdm import tqdm

import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
# import mpl_toolkits.mplot3d  # noqa: F401
# from matplotlib import ticker

from sklearn import datasets, manifold

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected

def main(n_graphs, graphs, features, full_graphs):
    print("hallo")
    class MyGraphDataset(InMemoryDataset):
        def __init__(self, root: str, n_graphs: int, graphs, features,
                    transform=None, pre_transform=None):
            """
            root      : dossier racine pour stocker raw/processed
            n_graphs  : nombre total de graphes à charger via get_graph()
            """
            self.n_graphs = n_graphs
            self.graphs = graphs
            self.features = features
            super().__init__(root, transform, pre_transform)
            # Charge les données traitées
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        @property
        def raw_file_names(self):
            # pas de fichiers à télécharger
            return []

        @property
        def processed_file_names(self):
            # nom du fichier torch.save
            return ['data.pt']

        def download(self):
            # inutile si tout est généré en local
            pass

        def process(self):
            data_list = []

            for idx in range(self.n_graphs):
                # Récupère la liste d’arêtes (liste de couples d’indices)
                edge_list = graphs[idx]
                
                # Passe en tenseur PyTorch (2, num_edges)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
                # Si graphe non orienté, duplique les arêtes
                edge_index = to_undirected(edge_index)

                feats = features[idx]
                print(feats)
                print(edge_list)
                # Conversion en tenseur float
                x = torch.reshape(torch.tensor(feats, dtype=torch.float), (len(feats), 1))
                # Crée l’objet Data
                data = Data(edge_index=edge_index, x=x)
                
                # Optionnel : spécifier num_nodes si nécessaire
                data.num_nodes = x.size(0)

                data_list.append(data)

            # Collate + sauvegarde sur disque
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    name = "small_graphs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "data/small_graphs"
    datasets = MyGraphDataset(path, n_graphs, graphs, features)

    print("Nombre de graphes :", len(datasets))
    print("Shape des features du graphe 0 :", datasets[0].x.shape)

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.convs = torch.nn.ModuleList(
                [
                    SAGEConv(in_channels, hidden_channels),
                    SAGEConv(hidden_channels, hidden_channels),
                    SAGEConv(hidden_channels, hidden_channels),
                ]
            )

            self.activations = torch.nn.ModuleList()
            self.activations.extend(
                [
                    torch.nn.PReLU(hidden_channels),
                    torch.nn.PReLU(hidden_channels),
                    torch.nn.PReLU(hidden_channels),
                ]
            )

        def forward(self, x, edge_index, batch_size):
            for conv, act in zip(self.convs, self.activations):
                x = conv(x, edge_index)
                x = act(x)
            return x[:batch_size]

        def infer(self, x, edge_index):
            for conv, act in zip(self.convs, self.activations):
                x = conv(x, edge_index)
                x = act(x)

            return x
            # return x[edge_index[0]] * x[edge_index[1]]


    def corruption(x, edge_index, batch_size):
        return x[torch.randperm(x.size(0))], edge_index, batch_size


    encoder = Encoder(datasets.num_features, 32)
    model = DeepGraphInfomax(
        hidden_channels=32,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    def train(epoch, train_loader):
        model.train()

        total_loss = total_examples = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            batch.to(device)
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(batch.x, batch.edge_index, batch.batch_size)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)

        return total_loss / total_examples


    def main2():
        train_loader = DataLoader(datasets, batch_size=32, shuffle=True)

        ###
        # Training
        ###
        for epoch in range(1, 32):
            loss = train(epoch, train_loader)
            print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

        ###
        # Testing
        ###
        print("*" * 20, "TESTING", "*" * 20)
        data = datasets[0].to(device)
        full_graph = full_graphs[0]
        print("date edge_index", data.edge_index)
        print("data nodes", data.num_nodes)
        print("data nb_edge ", data.num_edges)
        print("data nb_features ", data.num_features)


        # Compute the edge embeddings for each edge of the graph.
        edge_embs = encoder.infer(data.x, data.edge_index)
        print("embs", edge_embs.size())
        emb_norm = F.normalize(edge_embs, p=2, dim=1)
        import math
        def get_similarities(tree_edges, tree_features):
                
            # Passe en tenseur PyTorch (2, num_edges)
            edge_index = torch.tensor(tree_edges, dtype=torch.long).t().contiguous()
            
            # Si graphe non orienté, duplique les arêtes
            edge_index = to_undirected(edge_index)
            #print(edge_index.size())
            # Conversion en tenseur float
            x = torch.reshape(torch.tensor(tree_features, dtype=torch.float), (len(tree_features), 1))

            edge_embs = encoder.infer(x, edge_index)
            #print("embs", edge_embs.size())
            emb_norm = F.normalize(edge_embs, p=2, dim=1)

            edges = []
            sim_edges = []
            for [u, v] in full_graph:
                edges.append((u, v))
                sim_edges.append(1.0)# float(torch.dot(emb_norm[u], emb_norm[v])))

            # t_sne = manifold.TSNE(
            #     n_components=2,
            #     perplexity=30,
            #     init="random",
            #     max_iter=250,
            #     random_state=0,
            # )
            # S_t_sne = t_sne.fit_transform(emb_norm.detach().numpy())

            # def add_2d_scatter(ax, points, points_color, title=None):
            #     x, y = points.T
            #     ax.scatter(x, y, s=50, alpha=0.8)
            #     ax.set_title(title)
            #     # ax.xaxis.set_major_formatter(ticker.NullFormatter())
            #     # ax.yaxis.set_major_formatter(ticker.NullFormatter())

            # def plot_2d(points, points_color, title):
            #     fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
            #     fig.suptitle(title, size=16)
            #     add_2d_scatter(ax, points, points_color)
            #     plt.show()
            
            # plot_2d(S_t_sne, None, "T-distributed Stochastic  \n Neighbor Embedding")

            return edges, sim_edges

        edges = []
        sim_edges = []
        for [u, v] in full_graph:
            edges.append((u, v))
            sim_edges.append( float(torch.dot(emb_norm[u], emb_norm[v])))

        sim_mat = torch.mm(emb_norm, emb_norm.t())
        sim_mat.fill_diagonal_(-float("inf"))

        print("similarity size: ", sim_mat.size())
        print("similarity matrix", sim_mat)

        values, indices = torch.topk(sim_mat.flatten(), k=20)
        print("values topk: ", values)
        print("indices topk: ", indices)
        r_indices = indices // data.num_nodes
        c_indices = indices % data.num_nodes

        print("Most similar edge pairs:")
        for i in range(r_indices.size(0)):
            print(
                f"edge {r_indices[i].item()}, {c_indices[i].item()}: {sim_mat[r_indices[i], c_indices[i]].item()}"
            )

        # print(edges)
        # print(sim_edges)
        return get_similarities


    return main2()


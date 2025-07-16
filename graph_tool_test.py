import graph_tool.all as gt
import time
print("hello from python")


def count_each(l):
    d = dict()
    for x in l:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1

    return d

def find_communities(edges, kmin, kmax):
    print("debut fonction")
    g = gt.Graph(edges, directed=False)
    print("fin importation")

    t1 = time.perf_counter()
    state = gt.minimize_blockmodel_dl(g, state=gt.ModularityState, multilevel_mcmc_args=dict(B_min=kmin, B_max=kmax))
    print("louvain", time.perf_counter() - t1)
    blocks = list(state.b)
    # print(state.get_nonempty_B())
    # print(state.get_N())
    # state.draw(output="polbooks_louvain_mdl.svg")
    ce = count_each(blocks)
    print(ce)
    print(len(ce))
    return blocks

def main2():

    g = gt.collection.data["power"]

    state = gt.minimize_blockmodel_dl(g, multilevel_mcmc_args=dict(B_min=100, B_max=100))
    print(state.get_nonempty_B())
    print(state.get_N())
    state.draw(pos=g.vp["pos"], vertex_shape=state.get_blocks(), output="polbooks_blocks_mdl.svg")

    block_set = set()
    for bi in state.get_blocks():
        block_set.add(bi)

    block_dict = {bi:i for (i, bi) in enumerate(block_set)}

    print([block_dict[bi] for bi in state.get_blocks()])
def main():
    print("hello")
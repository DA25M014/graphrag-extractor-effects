"""Read a GraphRAG output directory into a plain edge list.

GraphRAG writes parquet tables (entities, relationships, communities).
`relationships.source` and `.target` are entity *titles*, not ids, so the
adapter maps titles to contiguous vertex indices itself.

The graph this returns is the graph GraphRAG actually clusters, so that
structural statistics describe the object the pipeline works on rather
than a looser reading of the same tables. GraphRAG calls
`graspologic_native.hierarchical_leiden` with `use_lcc=True` by default,
so we take the largest connected component; it treats the graph as
simple and undirected, so we drop self-loops and collapse duplicate
pairs. Edge weights are discarded because this study compares topology
across extractors, and the weights are not comparable between the NLP
and LLM paths.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

ENTITIES = 'entities.parquet'
RELATIONSHIPS = 'relationships.parquet'
COMMUNITIES = 'communities.parquet'

# GraphRAG's own clustering defaults, recorded for reference.
GRAPHRAG_DEFAULTS = {'max_cluster_size': 10, 'resolution': 1.0,
                     'randomness': 0.001, 'seed': 0xDEADBEEF,
                     'use_lcc': True, 'use_modularity': True}


def _find(out_dir, name):
    """Locate a table, tolerating the pre-2.0 `create_final_*` names."""
    direct = os.path.join(out_dir, name)
    if os.path.exists(direct):
        return direct
    legacy = os.path.join(out_dir, 'create_final_' + name)
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(
        f"{name} not found in {out_dir} (tried {name} and "
        f"create_final_{name}); is this a GraphRAG output directory?")


def load_entity_graph(out_dir, verbose=True):
    """GraphRAG output directory -> (E, n, meta).

    E is an (m, 2) int64 array of undirected edges on vertices 0..n-1,
    simple and self-loop free, restricted to the largest connected
    component. meta records what was dropped, so the gap between the
    entities an extractor emitted and the graph they actually form can
    be reported rather than hidden.
    """
    ents = pd.read_parquet(_find(out_dir, ENTITIES))
    rels = pd.read_parquet(_find(out_dir, RELATIONSHIPS))

    titles = ents['title'].astype(str).tolist()
    idx = {t: i for i, t in enumerate(titles)}
    n_raw = len(titles)

    src = rels['source'].astype(str)
    dst = rels['target'].astype(str)
    dangling = int((~src.isin(idx)).sum() + (~dst.isin(idx)).sum())

    pairs = []
    for a, b in zip(src, dst):
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue          # relationship naming an unextracted entity
        if ia == ib:
            continue          # self-loop
        pairs.append((ia, ib) if ia < ib else (ib, ia))

    n_selfloops = int(sum(1 for a, b in zip(src, dst) if idx.get(a)
                          is not None and idx.get(a) == idx.get(b)))
    uniq = sorted(set(pairs))
    n_dupes = len(pairs) - len(uniq)

    G = nx.Graph()
    G.add_nodes_from(range(n_raw))
    G.add_edges_from(uniq)

    if G.number_of_edges() == 0:
        raise ValueError(f"no usable relationships in {out_dir}")

    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc).copy()
    relabel = {v: i for i, v in enumerate(sorted(H.nodes()))}
    E = np.array([(relabel[u], relabel[v]) for u, v in H.edges()],
                 dtype=np.int64)
    n = H.number_of_nodes()

    deg = np.asarray([d for _, d in H.degree()], dtype=float)
    meta = {
        'n_entities_raw': n_raw,
        'n_relationships_raw': int(len(rels)),
        'n_dangling_endpoints': dangling,
        'n_selfloops_dropped': n_selfloops,
        'n_duplicate_pairs_dropped': n_dupes,
        'n': n,
        'm': int(E.shape[0]),
        'n_dropped_off_lcc': n_raw - n,
        'dbar': float(2 * E.shape[0] / n),
        'dmax': int(deg.max()),
        'tail_index_hill': _hill(deg),
        'lcc_fraction': n / n_raw,
        # keep the title order so leaf memberships stay interpretable
        'titles': [titles[v] for v in sorted(H.nodes())],
    }
    if verbose:
        print(f"[adapter] {out_dir}: n={n} m={meta['m']} "
              f"dbar={meta['dbar']:.2f} dmax={meta['dmax']} "
              f"Hill_alpha={meta['tail_index_hill']:.2f} "
              f"(dropped {meta['n_dropped_off_lcc']} off-LCC entities, "
              f"{n_dupes} duplicate pairs, {n_selfloops} self-loops)")
    return E, n, meta


def _hill(deg, frac=0.1):
    """Hill tail-index estimate on the top `frac` of degrees.

    Reported because degree heterogeneity is one of the ways these
    graphs differ from each other, and a hub-dominated graph behaves
    differently from an even one under any downstream method. Smaller
    alpha means a heavier tail.
    """
    d = np.sort(deg[deg > 0])[::-1]
    k = max(5, int(len(d) * frac))
    k = min(k, len(d) - 1)
    if k < 5:
        return float('nan')
    top = d[:k]
    return float(1.0 / np.mean(np.log(top / d[k])))


def load_communities(out_dir):
    """GraphRAG's own hierarchy: per-level community counts.

    Returns dict(level -> dict(n_communities, sizes)) plus `total`,
    the number of community reports GraphRAG would have paid an LLM to
    write (one per community at every level).
    """
    comms = pd.read_parquet(_find(out_dir, COMMUNITIES))
    by_level = {}
    for lvl, grp in comms.groupby('level'):
        sizes = (grp['size'].tolist() if 'size' in grp.columns
                 else [len(e) for e in grp['entity_ids']])
        by_level[int(lvl)] = {'n_communities': int(len(grp)),
                              'sizes': sizes}
    return {'by_level': by_level,
            'total_communities': int(len(comms)),
            'n_levels': len(by_level)}

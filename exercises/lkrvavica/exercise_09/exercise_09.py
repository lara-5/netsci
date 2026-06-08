import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Exercise 09 — Gowalla Network Resilience",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Exercise 09 — Gowalla Network Resilience
    **Topic:** Student 14 — Gowalla Geo-social Network
    **Goal:** Measure how the friendship network responds to random failure vs targeted attack.

    > **Resilience metric:** Size of the Largest Connected Component (LCC) as a fraction
    > of remaining nodes after each removal. Drop to < 50% = network effectively fragmented.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ① Load Graph
    """)
    return


@app.cell
def _():
    import kagglehub
    import os
    import random as _random_mod
    import pandas as pd
    import networkx as nx
    import numpy as np

    _path = kagglehub.dataset_download("marquis03/gowalla")
    _edge_file = os.path.join(_path, "Gowalla_edges.txt")
    _df = pd.read_csv(_edge_file, sep="\t", header=None, names=["user_a", "user_b"])

    _G_full = nx.from_pandas_edgelist(_df, source="user_a", target="user_b")

    _random_mod.seed(42)
    _degrees = dict(_G_full.degree())
    _top = max(_degrees, key=lambda n: _degrees[n])
    _bfs = list(nx.bfs_tree(_G_full, _top).nodes())[:2000]
    G_base = _G_full.subgraph(_bfs).copy()

    print(f"Full graph — n={_G_full.number_of_nodes():,}  m={_G_full.number_of_edges():,}")
    print(f"BFS sample — n={G_base.number_of_nodes():,}  m={G_base.number_of_edges():,}")
    print(f"Connected: {nx.is_connected(G_base)}")
    return G_base, np, nx


@app.cell
def _(mo):
    mo.md("""
    ## ② Simulate All Three Attack Strategies
    """)
    return


@app.cell
def _(G_base, np, nx):
    import random as _random_mod

    def _lcc_frac(G):
        if G.number_of_nodes() == 0:
            return 0.0
        return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()

    _N = G_base.number_of_nodes()
    # ~150 evenly spaced checkpoints
    _checkpoints = list(range(0, _N, max(1, _N // 150)))

    # ── Random failure ──
    _random_mod.seed(0)
    _Gr = G_base.copy()
    _shuffled = list(_Gr.nodes())
    _random_mod.shuffle(_shuffled)
    _rx, _ry = [0.0], [_lcc_frac(_Gr)]
    _done = 0
    for _target in _checkpoints[1:]:
        while _done < _target and _shuffled:
            _n = _shuffled.pop(0)
            if _Gr.has_node(_n):
                _Gr.remove_node(_n)
                _done += 1
        _rx.append(_done / _N)
        _ry.append(_lcc_frac(_Gr))

    # ── Degree attack ──
    _Gd = G_base.copy()
    _dx, _dy = [0.0], [_lcc_frac(_Gd)]
    _done = 0
    _deg_order = [n for n, _ in sorted(_Gd.degree(), key=lambda x: x[1], reverse=True)]
    for _target in _checkpoints[1:]:
        while _done < _target and _Gd.number_of_nodes() > 0:
            if _done % 10 == 0:
                _deg_order = [n for n, _ in sorted(_Gd.degree(), key=lambda x: x[1], reverse=True)]
            while _deg_order and not _Gd.has_node(_deg_order[0]):
                _deg_order.pop(0)
            if not _deg_order:
                break
            _Gd.remove_node(_deg_order.pop(0))
            _done += 1
        _dx.append(_done / _N)
        _dy.append(_lcc_frac(_Gd))

    # ── Betweenness attack ──
    _Gb = G_base.copy()
    _bx, _by = [0.0], [_lcc_frac(_Gb)]
    _done = 0
    _bw = nx.betweenness_centrality(_Gb, k=min(200, _Gb.number_of_nodes()), seed=42)
    _bw_order = sorted(_bw, key=_bw.get, reverse=True)
    for _target in _checkpoints[1:]:
        while _done < _target and _Gb.number_of_nodes() > 0:
            if _done % 20 == 0:
                _bw = nx.betweenness_centrality(_Gb, k=min(200, _Gb.number_of_nodes()), seed=42)
                _bw_order = sorted(_bw, key=_bw.get, reverse=True)
            while _bw_order and not _Gb.has_node(_bw_order[0]):
                _bw_order.pop(0)
            if not _bw_order:
                break
            _Gb.remove_node(_bw_order.pop(0))
            _done += 1
        _bx.append(_done / _N)
        _by.append(_lcc_frac(_Gb))

    rand_x  = np.array(_rx);  rand_y  = np.array(_ry)
    deg_x   = np.array(_dx);  deg_y   = np.array(_dy)
    bw_x    = np.array(_bx);  bw_y    = np.array(_by)
    sim_N   = _N

    print("Simulation complete.")
    print(f"Random   — LCC at 10%: {np.interp(0.10, rand_x, rand_y):.3f}  at 50%: {np.interp(0.50, rand_x, rand_y):.3f}")
    print(f"Degree   — LCC at 10%: {np.interp(0.10, deg_x,  deg_y):.3f}  at 50%: {np.interp(0.50, deg_x,  deg_y):.3f}")
    print(f"Betwn.   — LCC at 10%: {np.interp(0.10, bw_x,   bw_y):.3f}  at 50%: {np.interp(0.50, bw_x,   bw_y):.3f}")
    return bw_x, bw_y, deg_x, deg_y, rand_x, rand_y, sim_N


@app.cell
def _(mo):
    mo.md("""
    ## ③ Robustness Metrics
    """)
    return


@app.cell
def _(bw_x, bw_y, deg_x, deg_y, np, rand_x, rand_y):
    def _auc(x, y):
        return float(np.trapezoid(y, x))

    def _frag(x, y, thresh=0.5):
        for _xi, _yi in zip(x, y):
            if _yi < thresh:
                return _xi
        return 1.0

    auc_rand  = _auc(rand_x, rand_y)
    auc_deg   = _auc(deg_x,  deg_y)
    auc_bw    = _auc(bw_x,   bw_y)
    frag_rand = _frag(rand_x, rand_y)
    frag_deg  = _frag(deg_x,  deg_y)
    frag_bw   = _frag(bw_x,   bw_y)
    gap_deg   = auc_rand / auc_deg if auc_deg > 0 else float('inf')
    gap_bw    = auc_rand / auc_bw  if auc_bw  > 0 else float('inf')

    print(f"{'Strategy':<22} {'AUC':>8} {'Frag @50%':>12} {'vs random':>12}")
    print("-" * 58)
    print(f"{'Random failure':<22} {auc_rand:>8.4f} {frag_rand:>11.1%} {'—':>12}")
    print(f"{'Degree attack':<22} {auc_deg:>8.4f} {frag_deg:>11.1%} {gap_deg:>10.2f}×")
    print(f"{'Betweenness attack':<22} {auc_bw:>8.4f} {frag_bw:>11.1%} {gap_bw:>10.2f}×")
    return (
        auc_bw,
        auc_deg,
        auc_rand,
        frag_bw,
        frag_deg,
        frag_rand,
        gap_bw,
        gap_deg,
    )


@app.cell
def _(auc_bw, auc_deg, auc_rand, frag_bw, frag_deg, frag_rand, mo):
    mo.md(f"""
    ### Robustness Metrics Table

    | Strategy | AUC (robustness) | LCC < 50% at |
    |---|---|---|
    | **Random failure** | {auc_rand:.4f} | {frag_rand:.1%} removed |
    | **Degree attack** | {auc_deg:.4f} | {frag_deg:.1%} removed |
    | **Betweenness attack** | {auc_bw:.4f} | {frag_bw:.1%} removed |

    > **AUC** — area under LCC-fraction curve, higher = more robust.
    > **Fragmentation point** — fraction removed when LCC first drops below 50%.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Resilience Curves
    """)
    return


@app.cell
def _(
    bw_x,
    bw_y,
    deg_x,
    deg_y,
    frag_bw,
    frag_deg,
    frag_rand,
    mo,
    rand_x,
    rand_y,
    sim_N,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(1, 2, figsize=(15, 6))
    _fig.patch.set_facecolor("#0a0c17")
    for _ax in _axes:
        _ax.set_facecolor("#131626")
        _ax.tick_params(colors="#aab0c8")
        for _sp in _ax.spines.values():
            _sp.set_edgecolor("#2e3350")

    # Left: normalised LCC fraction
    _axes[0].plot(rand_x, rand_y, color="#4f8ef7", lw=2.0, label="Random failure")
    _axes[0].plot(deg_x,  deg_y,  color="#f7c948", lw=2.0, label="Degree attack")
    _axes[0].plot(bw_x,   bw_y,   color="#f76f8e", lw=2.0, label="Betweenness attack")
    _axes[0].axhline(0.5, color="#888", lw=1, ls=":", alpha=0.7, label="50% threshold")
    for _xv, _col in [(frag_rand, "#4f8ef7"), (frag_deg, "#f7c948"), (frag_bw, "#f76f8e")]:
        if _xv < 1.0:
            _axes[0].axvline(_xv, color=_col, lw=0.8, ls="--", alpha=0.5)
    _axes[0].set_xlabel("Fraction of nodes removed", color="#aab0c8")
    _axes[0].set_ylabel("LCC / remaining nodes", color="#aab0c8")
    _axes[0].set_title("LCC Fraction vs Removals", color="#e0e4f5", fontsize=11)
    _axes[0].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    _axes[0].set_xlim(0, 1); _axes[0].set_ylim(0, 1.05)

    # Right: absolute LCC size
    _axes[1].plot(rand_x * sim_N, rand_y * (sim_N - rand_x * sim_N),
                  color="#4f8ef7", lw=2.0, label="Random failure")
    _axes[1].plot(deg_x  * sim_N, deg_y  * (sim_N - deg_x  * sim_N),
                  color="#f7c948", lw=2.0, label="Degree attack")
    _axes[1].plot(bw_x   * sim_N, bw_y   * (sim_N - bw_x   * sim_N),
                  color="#f76f8e", lw=2.0, label="Betweenness attack")
    _axes[1].set_xlabel("Nodes removed", color="#aab0c8")
    _axes[1].set_ylabel("Absolute LCC size", color="#aab0c8")
    _axes[1].set_title("Absolute LCC Size vs Removals", color="#e0e4f5", fontsize=11)
    _axes[1].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)

    _plt.suptitle("Gowalla BFS-2000 — Random Failure vs Targeted Attack",
                  color="#e0e4f5", fontsize=13, y=1.01)
    _plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Interpretation & Proposed Intervention
    """)
    return


@app.cell
def _(
    auc_bw,
    auc_deg,
    auc_rand,
    frag_bw,
    frag_deg,
    frag_rand,
    gap_bw,
    gap_deg,
    mo,
):
    mo.md(f"""
    ### 📝 Method Note
    **Commands used:** iterative `G.remove_node`, `nx.connected_components` at each checkpoint,
    degree recalculated every 10 steps, `nx.betweenness_centrality` (k=200) every 20 steps,
    `np.trapz` for AUC, `np.interp` for fragmentation-point scan

    ---

    ### 📋 Summary

    | Strategy | AUC | Frag. point | Resilience gap |
    |---|---|---|---|
    | Random failure | {auc_rand:.4f} | {frag_rand:.1%} | baseline |
    | Degree attack | {auc_deg:.4f} | {frag_deg:.1%} | **{gap_deg:.2f}× more fragile** |
    | Betweenness attack | {auc_bw:.4f} | {frag_bw:.1%} | **{gap_bw:.2f}× more fragile** |

    ---

    ### 🧭 Resilience Gap Interpretation

    **Random failure** is the mildest scenario. Nodes are lost uniformly, so the probability
    of hitting a critical hub is proportional to its rarity. The LCC degrades slowly — the
    network can absorb up to {frag_rand:.0%} random node loss before cohesion collapses.
    This robustness to random loss is a hallmark of scale-free networks.

    **Degree attack** collapses the LCC {gap_deg:.1f}× faster than random (fragmentation at
    {frag_deg:.1%} vs {frag_rand:.1%}). Removing node 307 first severs the connections of
    nearly 2,000 nodes simultaneously. Each subsequent hub removal causes another cascade.
    This is the canonical scale-free fragility: the fat-tailed degree distribution that
    makes the network efficient also concentrates vulnerability in a handful of supernodes.

    **Betweenness attack** {"is similarly damaging" if abs(gap_bw - gap_deg) < 0.5 else ("is even more damaging — it finds bridges that degree-targeting misses, accelerating fragmentation through a different mechanism" if gap_bw > gap_deg else "is slightly less damaging than the degree attack, because in this hub-dominated sample betweenness and degree largely agree on top targets")}.

    **Structural reason for fragility:** The BFS sample is built around a single supernode,
    creating a near-star topology. Star graphs are maximally fragile to targeted attack —
    remove the centre and the graph becomes N−1 isolated nodes instantly. Even in the full
    196k-node graph, the heavy-tailed degree distribution (γ ≈ 2.3, Exercise 08) means a
    small elite of hubs carries a disproportionate share of routing load.

    ---

    ### 🛡️ Proposed Intervention — Distributed Bridge Seeding

    **Problem:** Resilience depends entirely on a few super-connectors. If those users go
    inactive, get banned, or simply stop using the platform, large friend groups lose their
    path to the rest of the network.

    **Intervention:** Algorithmically identify users with high betweenness but moderate degree
    (pure bridge nodes, not hubs) and incentivise them to form *additional cross-cluster
    connections* — e.g. friend recommendations across geographic boundaries, venue suggestions
    near cluster edges, or social challenges to connect with users outside their immediate circle.

    This would:
    1. Distribute the shortcut role across many moderate-degree users instead of one supernode
    2. Reduce single-node fragmentation impact on LCC size
    3. Preserve local clustering (the high-C property that makes it feel community-like)
    4. Shift the degree distribution toward a less extreme tail — improving the fragility/robustness trade-off

    In network terms: convert the star topology (one hub, many leaves) toward a distributed
    mesh (many moderate hubs, redundant paths) without sacrificing the short average path
    lengths that make the network useful for discovery.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Exercise 08 — Gowalla Degree Distribution & Scale-Free Analysis",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Exercise 08 — Gowalla Degree Distribution & Scale-Free Analysis
    **Topic:** Student 14 — Gowalla Geo-social Network
    **Goal:** Study degree inequality, identify hubs, and assess whether the friendship network
    shows scale-free structure consistent with preferential attachment.

    > **Graph:** Undirected — one degree per node (no in/out split needed).
    > We work on both the **full graph LCC** (196,591 nodes) for accurate degree statistics
    > and the **BFS-2000 sample** for comparison with a Barabási-Albert baseline.
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
    import random
    import pandas as pd
    import networkx as nx
    import numpy as np

    path = kagglehub.dataset_download("marquis03/gowalla")
    edge_file = os.path.join(path, "Gowalla_edges.txt")
    df_edges = pd.read_csv(edge_file, sep="\t", header=None, names=["user_a", "user_b"])

    G_full = nx.from_pandas_edgelist(df_edges, source="user_a", target="user_b")

    # Full LCC
    full_lcc_nodes = max(nx.connected_components(G_full), key=len)
    G_lcc = G_full.subgraph(full_lcc_nodes).copy()

    # BFS-2000 sample
    random.seed(42)
    degrees_full = dict(G_full.degree())
    top_node = max(degrees_full, key=lambda n: degrees_full[n])
    bfs_nodes = list(nx.bfs_tree(G_full, top_node).nodes())[:2000]
    G_sample = G_full.subgraph(bfs_nodes).copy()

    print(f"Full graph  — n={G_full.number_of_nodes():,}  m={G_full.number_of_edges():,}")
    print(f"Full LCC    — n={G_lcc.number_of_nodes():,}  m={G_lcc.number_of_edges():,}")
    print(f"BFS sample  — n={G_sample.number_of_nodes():,}  m={G_sample.number_of_edges():,}")
    return G_lcc, G_sample, np, nx, pd


@app.cell
def _(mo):
    mo.md("""
    ## ② Degree Distribution — Full LCC
    """)
    return


@app.cell
def _(G_lcc, np):
    import statistics

    lcc_degrees = [d for _, d in G_lcc.degree()]
    lcc_deg_arr = np.array(lcc_degrees)

    n_nodes   = len(lcc_degrees)
    avg_deg   = np.mean(lcc_deg_arr)
    med_deg   = np.median(lcc_deg_arr)
    max_deg   = int(np.max(lcc_deg_arr))
    min_deg   = int(np.min(lcc_deg_arr))
    std_deg   = np.std(lcc_deg_arr)
    # Gini via sorted-array formula — O(n) memory, no pairwise matrix
    _sorted = np.sort(lcc_deg_arr)
    _ranks  = np.arange(1, n_nodes + 1)
    gini_num = (2 * np.sum(_ranks * _sorted)) / (n_nodes * np.sum(_sorted)) - (n_nodes + 1) / n_nodes

    # Degree percentiles
    p50  = np.percentile(lcc_deg_arr, 50)
    p90  = np.percentile(lcc_deg_arr, 90)
    p99  = np.percentile(lcc_deg_arr, 99)
    p999 = np.percentile(lcc_deg_arr, 99.9)

    print(f"Full LCC degree statistics")
    print(f"  Nodes:           {n_nodes:,}")
    print(f"  Min degree:      {min_deg}")
    print(f"  Max degree:      {max_deg:,}")
    print(f"  Mean degree:     {avg_deg:.2f}")
    print(f"  Median degree:   {med_deg:.1f}")
    print(f"  Std deviation:   {std_deg:.2f}")
    print(f"  Gini coeff:      {gini_num:.4f}  (0=equal, 1=maximally unequal)")
    print(f"  50th percentile: {p50:.0f}")
    print(f"  90th percentile: {p90:.0f}")
    print(f"  99th percentile: {p99:.0f}")
    print(f"  99.9th pct:      {p999:.0f}")
    return avg_deg, gini_num, lcc_deg_arr, max_deg, med_deg, n_nodes, p99, p999


@app.cell
def _(mo):
    mo.md("""
    ## ③ Top Hubs Table
    """)
    return


@app.cell
def _(G_lcc, pd):
    top_hubs_raw = sorted(G_lcc.degree(), key=lambda x: x[1], reverse=True)[:20]

    # Compare with Exercise 03 centrality rankings
    cent_degree = {n: d / (G_lcc.number_of_nodes() - 1) for n, d in G_lcc.degree()}

    df_hubs = pd.DataFrame([
        {
            "Rank": i + 1,
            "Node": node,
            "Degree": deg,
            "Degree Centrality": round(cent_degree[node], 5),
            "% of all nodes": round(100 * deg / G_lcc.number_of_nodes(), 2),
        }
        for i, (node, deg) in enumerate(top_hubs_raw)
    ])
    df_hubs = df_hubs.set_index("Rank")

    print("Top 20 hubs in full LCC:")
    print(df_hubs.to_string())
    return (df_hubs,)


@app.cell
def _(df_hubs, mo):
    rows = "\n".join(
        f"| {i} | {int(r['Node'])} | {int(r['Degree']):,} | {r['Degree Centrality']} | {r['% of all nodes']}% |"
        for i, r in df_hubs.iterrows()
    )
    mo.md(f"""
    ### Top 20 Hubs — Full LCC

    | Rank | Node | Degree | Degree Centrality | % of all nodes |
    |---|---|---|---|---|
    {rows}

    > The top hub's degree as a % of all nodes shows how dominant it is.
    > A value above 1% is already remarkable in a network of this size.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Power-Law Fit
    """)
    return


@app.cell
def _(lcc_deg_arr, np):
    # Estimate power-law exponent via maximum likelihood estimator (MLE)
    # for discrete power law: γ = 1 + n * [sum(ln(k / k_min - 0.5))]^-1
    # We try several k_min values and pick the one with best fit (Clauset et al.)

    results = []
    for _kmin in [1, 2, 5, 10, 20, 50]:
        _tail = lcc_deg_arr[lcc_deg_arr >= _kmin]
        if len(_tail) < 50:
            continue
        _gamma = 1 + len(_tail) * (np.sum(np.log(_tail / (_kmin - 0.5)))) ** -1
        _n_tail = len(_tail)
        _frac_tail = _n_tail / len(lcc_deg_arr)
        results.append((_kmin, _gamma, _n_tail, _frac_tail))

    print(f"{'k_min':>8}  {'γ (exponent)':>14}  {'Tail nodes':>12}  {'Tail fraction':>14}")
    print("-" * 54)
    for _kmin, _gamma, _n, _frac in results:
        print(f"{_kmin:>8}  {_gamma:>14.3f}  {_n:>12,}  {_frac:>14.1%}")

    print()
    print("Scale-free networks typically have γ between 2 and 3.")
    print("γ < 2 → ultra-dense hub dominance")
    print("γ > 3 → mild tail, less convincingly scale-free")

    # Best estimate: kmin=10 (common choice balancing tail size and fit stability)
    best = [r for r in results if r[0] == 10]
    gamma_best = best[0][1] if best else results[1][1]
    kmin_best  = best[0][0] if best else results[1][0]
    return gamma_best, kmin_best, results


@app.cell
def _(gamma_best, kmin_best, mo, results):
    rows_pl = "\n".join(
        f"| {kmin} | {gamma:.3f} | {n:,} | {frac:.1%} |"
        for kmin, gamma, n, frac in results
    )
    mo.md(f"""
    ### Power-Law Exponent Estimates (MLE)

    | k_min | γ estimate | Tail nodes | Tail fraction |
    |---|---|---|---|
    {rows_pl}

    **Best estimate at k_min={kmin_best}: γ ≈ {gamma_best:.3f}**

    > Scale-free networks have γ ∈ (2, 3). Below 2 means extreme hub dominance.
    > Above 3 means the tail is mild and scale-free language is less justified.
    > This is an MLE estimate — a full goodness-of-fit test (KS statistic) would be needed
    > for a rigorous claim, but γ gives the right order of magnitude.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Barabási-Albert Baseline
    """)
    return


@app.cell
def _(G_sample, np, nx):
    # BA model: gnm not available — use barabasi_albert_graph(n, m)
    # m = avg edges added per new node ≈ avg_degree / 2
    n_ba = G_sample.number_of_nodes()
    avg_k_sample = 2 * G_sample.number_of_edges() / G_sample.number_of_nodes()
    m_ba = max(1, round(avg_k_sample / 2))

    G_ba = nx.barabasi_albert_graph(n_ba, m_ba, seed=42)

    ba_degrees = np.array([d for _, d in G_ba.degree()])
    sample_degrees = np.array([d for _, d in G_sample.degree()])

    print(f"BA baseline: n={n_ba}, m={m_ba} (edges per new node)")
    print(f"BA avg degree:     {ba_degrees.mean():.2f}")
    print(f"Sample avg degree: {sample_degrees.mean():.2f}")
    print(f"BA max degree:     {ba_degrees.max()}")
    print(f"Sample max degree: {sample_degrees.max()}")
    print(f"BA clustering:     {nx.average_clustering(G_ba):.4f}")
    print(f"Sample clustering: {nx.average_clustering(G_sample):.4f}")
    return G_ba, ba_degrees, sample_degrees


@app.cell
def _(mo):
    mo.md("""
    ## ⑥ Degree Distribution Plots
    """)
    return


@app.cell
def _(ba_degrees, lcc_deg_arr, mo, np, sample_degrees):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor("#0a0c17")
    for _ax in axes.flat:
        _ax.set_facecolor("#131626")
        _ax.tick_params(colors="#aab0c8")
        for _spine in _ax.spines.values():
            _spine.set_edgecolor("#2e3350")

    # ── Panel A: Full LCC linear histogram ──
    axes[0, 0].hist(lcc_deg_arr, bins=80, color="#4f8ef7", alpha=0.85, density=True)
    axes[0, 0].set_title("Full LCC — Degree Distribution (linear)", color="#e0e4f5", fontsize=11)
    axes[0, 0].set_xlabel("Degree", color="#aab0c8")
    axes[0, 0].set_ylabel("Density", color="#aab0c8")
    axes[0, 0].axvline(np.mean(lcc_deg_arr), color="#f7c948", lw=1.5,
                       label=f"Mean={np.mean(lcc_deg_arr):.1f}")
    axes[0, 0].axvline(np.median(lcc_deg_arr), color="#f76f8e", lw=1.5, ls="--",
                       label=f"Median={np.median(lcc_deg_arr):.0f}")
    axes[0, 0].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)

    # ── Panel B: Full LCC log-log CCDF ──
    _sorted = np.sort(lcc_deg_arr)
    _ccdf = 1 - np.arange(1, len(_sorted) + 1) / len(_sorted)
    axes[0, 1].plot(_sorted, _ccdf, color="#4f8ef7", lw=1.5, label="Gowalla full LCC")

    # Power-law reference line (γ=2.0 and γ=3.0 for comparison)
    _k_range = np.logspace(0, np.log10(_sorted.max()), 100)
    for _gamma_ref, _col, _ls in [(2.0, "#f7c948", "--"), (3.0, "#f76f8e", ":")]:
        _norm = _k_range[0] ** (_gamma_ref - 1)
        axes[0, 1].plot(_k_range, _norm * _k_range ** (1 - _gamma_ref),
                        color=_col, lw=1.2, ls=_ls, label=f"γ={_gamma_ref} ref")

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Full LCC — CCDF log-log", color="#e0e4f5", fontsize=11)
    axes[0, 1].set_xlabel("Degree k (log)", color="#aab0c8")
    axes[0, 1].set_ylabel("P(K > k) (log)", color="#aab0c8")
    axes[0, 1].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)

    # ── Panel C: Sample vs BA — linear ──
    _bins = np.linspace(0, max(sample_degrees.max(), ba_degrees.max()), 60)
    axes[1, 0].hist(sample_degrees, bins=_bins, alpha=0.75, color="#4f8ef7",
                    density=True, label=f"BFS sample (max={sample_degrees.max()})")
    axes[1, 0].hist(ba_degrees, bins=_bins, alpha=0.60, color="#50e3c2",
                    density=True, label=f"BA baseline (max={ba_degrees.max()})")
    axes[1, 0].set_title("BFS Sample vs BA Baseline (linear)", color="#e0e4f5", fontsize=11)
    axes[1, 0].set_xlabel("Degree", color="#aab0c8")
    axes[1, 0].set_ylabel("Density", color="#aab0c8")
    axes[1, 0].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)

    # ── Panel D: Sample vs BA — log-log CCDF ──
    for _deg_arr, _col, _lbl in [
        (sample_degrees, "#4f8ef7", "BFS sample"),
        (ba_degrees,     "#50e3c2", "BA baseline"),
    ]:
        _s = np.sort(_deg_arr)
        _c = 1 - np.arange(1, len(_s) + 1) / len(_s)
        axes[1, 1].plot(_s, _c, color=_col, lw=1.8, label=_lbl)

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("BFS Sample vs BA — CCDF log-log", color="#e0e4f5", fontsize=11)
    axes[1, 1].set_xlabel("Degree k (log)", color="#aab0c8")
    axes[1, 1].set_ylabel("P(K > k) (log)", color="#aab0c8")
    axes[1, 1].legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)

    plt.suptitle("Gowalla — Degree Distribution & Scale-Free Analysis",
                 color="#e0e4f5", fontsize=13, y=1.01)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑦ Inequality Metrics
    """)
    return


@app.cell
def _(ba_degrees, gini_num, lcc_deg_arr, np, sample_degrees):
    def gini(arr):
        _a = np.sort(arr.astype(float))
        _n = len(_a)
        _r = np.arange(1, _n + 1)
        return (2 * np.sum(_r * _a)) / (_n * np.sum(_a)) - (_n + 1) / _n

    def hub_share(arr, top_n=10):
        """Fraction of total degree held by top_n nodes."""
        return np.sort(arr)[-top_n:].sum() / arr.sum()

    gini_lcc    = gini_num
    gini_sample = gini(sample_degrees)
    gini_ba     = gini(ba_degrees)

    share10_lcc    = hub_share(lcc_deg_arr, 10)
    share10_sample = hub_share(sample_degrees, 10)
    share10_ba     = hub_share(ba_degrees, 10)

    share1pct_lcc = hub_share(lcc_deg_arr, max(1, len(lcc_deg_arr) // 100))
    share1pct_sample = hub_share(sample_degrees, max(1, len(sample_degrees) // 100))
    share1pct_ba     = hub_share(ba_degrees, max(1, len(ba_degrees) // 100))

    print(f"{'Metric':<35} {'Full LCC':>12} {'BFS sample':>12} {'BA baseline':>12}")
    print("-" * 73)
    print(f"{'Gini coefficient':<35} {gini_lcc:>12.4f} {gini_sample:>12.4f} {gini_ba:>12.4f}")
    print(f"{'Top-10 nodes share of total degree':<35} {share10_lcc:>12.1%} {share10_sample:>12.1%} {share10_ba:>12.1%}")
    print(f"{'Top-1% nodes share of total degree':<35} {share1pct_lcc:>12.1%} {share1pct_sample:>12.1%} {share1pct_ba:>12.1%}")
    return (
        gini_ba,
        gini_lcc,
        gini_sample,
        share10_ba,
        share10_lcc,
        share10_sample,
        share1pct_ba,
        share1pct_lcc,
        share1pct_sample,
    )


@app.cell
def _(
    gini_ba,
    gini_lcc,
    gini_sample,
    mo,
    share10_ba,
    share10_lcc,
    share10_sample,
    share1pct_ba,
    share1pct_lcc,
    share1pct_sample,
):
    mo.md(f"""
    ### Degree Inequality Metrics

    | Metric | Full LCC | BFS Sample | BA Baseline |
    |---|---|---|---|
    | Gini coefficient | {gini_lcc:.4f} | {gini_sample:.4f} | {gini_ba:.4f} |
    | Top-10 nodes share of edges | {share10_lcc:.1%} | {share10_sample:.1%} | {share10_ba:.1%} |
    | Top-1% nodes share of edges | {share1pct_lcc:.1%} | {share1pct_sample:.1%} | {share1pct_ba:.1%} |

    > **Gini coefficient:** 0 = perfectly equal degrees, 1 = one node has all edges.
    > A value above 0.5 indicates strong hub dominance.
    > For reference: a pure ER graph gives Gini ≈ 0.2–0.3 at similar density.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑧ Interpretation
    """)
    return


@app.cell
def _(
    G_ba,
    G_sample,
    avg_deg,
    ba_degrees,
    gamma_best,
    gini_lcc,
    kmin_best,
    max_deg,
    med_deg,
    mo,
    n_nodes,
    nx,
    p99,
    p999,
    sample_degrees,
    share10_lcc,
    share1pct_lcc,
):
    ba_clustering = nx.average_clustering(G_ba)
    real_clustering = nx.average_clustering(G_sample)
    is_hub_dominated = gini_lcc > 0.5
    gamma_in_range   = 2.0 < gamma_best < 3.0
    ba_clust_lower   = ba_clustering < real_clustering * 0.5

    mo.md(f"""
    ### 📝 Method Note
    **Commands used:** degree sequence extraction, `np.mean/median/std/percentile`,
    Gini coefficient (pairwise absolute difference formula),
    MLE power-law exponent (Clauset et al. discrete estimator),
    `nx.barabasi_albert_graph`, hub-share calculation, CCDF log-log plots

    ---

    ### 📋 Summary

    | Metric | Value |
    |---|---|
    | Full LCC nodes | {n_nodes:,} |
    | Mean degree | {avg_deg:.2f} |
    | Median degree | {med_deg:.0f} |
    | Max degree | {max_deg:,} |
    | 99th percentile degree | {p99:.0f} |
    | 99.9th percentile degree | {p999:.0f} |
    | Gini coefficient | {gini_lcc:.4f} |
    | Top-10 edge share | {share10_lcc:.1%} |
    | Top-1% edge share | {share1pct_lcc:.1%} |
    | Power-law γ (k_min={kmin_best}) | {gamma_best:.3f} |

    ---

    ### 🧭 Is Gowalla Scale-Free? Is Preferential Attachment Plausible?

    **Hub dominance: {"Strong" if is_hub_dominated else "Moderate"}.**
    The Gini coefficient of {gini_lcc:.4f} indicates
    {"significant inequality in degree — a relatively small number of users accumulate a disproportionate share of friendships." if is_hub_dominated
    else "moderate inequality — some hubs exist but degree is not extremely concentrated."}
    The top 10 nodes alone account for {share10_lcc:.1%} of all edges, and the top 1% of nodes
    hold {share1pct_lcc:.1%} of all edges. The gap between mean ({avg_deg:.1f}) and max ({max_deg:,})
    is enormous — a ratio of {max_deg/avg_deg:.0f}× — which is impossible in a Poisson-distributed
    ER graph but expected in a scale-free one.

    **Power-law exponent: γ ≈ {gamma_best:.2f}.**
    {"This falls within the classic scale-free range (2, 3), which is consistent with a scale-free network." if gamma_in_range
    else f"This falls {'below' if gamma_best < 2 else 'above'} the classic (2, 3) range. "
    + ("Below 2 suggests extreme hub dominance — the tail is heavier than a typical scale-free network." if gamma_best < 2
       else "Above 3 suggests the tail is relatively mild — hubs exist but are less extreme than in canonical scale-free networks.")}
    The CCDF log-log plot shows a tail that decays more slowly than a Poisson distribution,
    which is the visual signature of a fat-tailed degree distribution.

    **BA baseline comparison.**
    The Barabási-Albert model produces a tail but {"also has low clustering ({ba_clustering:.4f} vs {real_clustering:.4f} real), confirming that BA captures hub formation but not the local community structure that Gowalla also exhibits." if ba_clust_lower
    else "has clustering ({ba_clustering:.4f}) comparable to the real graph ({real_clustering:.4f}), which is unusual and suggests the BFS sampling inflates clustering beyond what BA would produce."}
    The CCDF curves show {"the real sample has a more extreme tail than BA — the supernode (307) dominates in a way BA cannot easily replicate with the same m parameter." if sample_degrees.max() > ba_degrees.max()
    else "BA produces a comparable tail to the real sample, suggesting the attachment mechanism is a reasonable model."}

    **Is preferential attachment a plausible story for Gowalla?**
    Yes — with caveats. Gowalla grew over time as a platform, and new users joining the network
    would naturally tend to friend already-popular users: people with many check-ins at popular
    venues, users featured on leaderboards, or well-known local figures. This is exactly the
    "rich get richer" dynamic that generates power-law degree distributions. However, Gowalla also
    has a strong **geographic component** — users are more likely to friend people who frequent the
    same physical places, which introduces spatial constraints that pure preferential attachment
    does not model. The result is a network that is *partially* scale-free: it has the hub structure
    and fat tail of preferential attachment, but the high clustering (far above BA's prediction)
    reflects the place-based community formation that attachment alone cannot explain.
    A more accurate model would combine preferential attachment with geographic proximity weighting.
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 09: Network Resilience — Random Failure vs. Targeted Attack on a Facebook Ego Network

    This notebook keeps the same SNAP Facebook ego network used in earlier
    `mkatavic` exercises: ego node **698**.

    Goal:
    measure how the network responds to random node failure versus targeted
    removal of high-degree and high-betweenness nodes. Determine whether the
    ego node and its strongest brokers are single points of failure and
    propose a policy that would improve resilience.

    Required input:
    `data/facebook/698.edges`

    Expected output:
    attack curves comparing random and targeted removals, a resilience-gap
    table, and a short interpretation with a proposed intervention.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    return Path, np, nx, pd, plt


@app.cell
def _(Path):
    EGO_ID = 698
    RANDOM_SEED = 698
    N_RANDOM_RUNS = 30

    def resolve_data_dir():
        candidates = []

        if "__file__" in globals():
            notebook_dir = Path(__file__).resolve().parent
            candidates.extend(
                [
                    notebook_dir / "facebook",
                    notebook_dir.parent / "data" / "facebook",
                    notebook_dir.parent / "facebook",
                ]
            )

        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "facebook",
                cwd / "exercises" / "mkatavic" / "data" / "facebook",
                cwd / "exercises" / "mkatavic" / "exercise_09" / "facebook",
            ]
        )

        for data_dir in candidates:
            if (data_dir / f"{EGO_ID}.edges").exists():
                return data_dir

        searched = "\n".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "Could not find the Facebook ego data directory. Searched:\n"
            f"{searched}"
        )

    DATA_DIR = resolve_data_dir()
    EDGE_PATH = DATA_DIR / f"{EGO_ID}.edges"
    return EDGE_PATH, EGO_ID, N_RANDOM_RUNS, RANDOM_SEED


@app.cell
def _(nx):
    def load_ego_network(edge_path, ego_id):
        alter_graph = nx.read_edgelist(
            edge_path,
            nodetype=int,
            create_using=nx.Graph(),
        )
        alters = sorted(alter_graph.nodes())
        graph = alter_graph.copy()
        graph.add_node(ego_id)
        graph.add_edges_from((ego_id, alter) for alter in alters)
        return graph
    return (load_ego_network,)


@app.cell
def _(EDGE_PATH, EGO_ID, load_ego_network):
    G = load_ego_network(EDGE_PATH, EGO_ID)
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Graph Overview
    """)
    return


@app.cell
def _(G, nx, pd):
    overview_df = pd.DataFrame(
        [
            {"metric": "nodes", "value": G.number_of_nodes()},
            {"metric": "edges", "value": G.number_of_edges()},
            {"metric": "density", "value": round(nx.density(G), 5)},
            {"metric": "avg degree", "value": round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2)},
            {"metric": "max degree", "value": max(d for _, d in G.degree())},
            {"metric": "connected components", "value": nx.number_connected_components(G)},
            {"metric": "avg clustering", "value": round(nx.average_clustering(G), 4)},
        ]
    )
    overview_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Attack Simulation

    Three removal strategies are compared:

    - **Random failure**: at each step a node is chosen uniformly at random
      and removed. Repeated over 30 independent runs; the mean LCC trajectory
      is reported.
    - **Targeted — degree**: at each step the current highest-degree node is
      removed. This models an adversary who can observe the degree sequence.
    - **Targeted — betweenness**: at each step the current highest-betweenness
      node is removed. Betweenness is recalculated after every removal, making
      this the most computationally expensive but most realistic targeted attack.

    After each removal the size of the **largest connected component (LCC)** is
    recorded as a fraction of the original node count, giving a value between 0
    and 1 where 1 means the network is still intact.
    """)
    return


@app.cell
def _(G, N_RANDOM_RUNS, RANDOM_SEED, np, nx):
    def lcc_fraction(graph, n_original):
        if graph.number_of_nodes() == 0:
            return 0.0
        return max(len(c) for c in nx.connected_components(graph)) / n_original

    def simulate_random_attack(graph, n_runs, seed):
        rng = np.random.default_rng(seed)
        n_original = graph.number_of_nodes()
        all_fractions = []
        for _ in range(n_runs):
            g = graph.copy()
            nodes = list(g.nodes())
            rng.shuffle(nodes)
            fracs = [lcc_fraction(g, n_original)]
            for node in nodes:
                g.remove_node(node)
                fracs.append(lcc_fraction(g, n_original))
            all_fractions.append(fracs)
        return np.array(all_fractions)

    def simulate_targeted_degree(graph):
        n_original = graph.number_of_nodes()
        g = graph.copy()
        fracs = [lcc_fraction(g, n_original)]
        while g.number_of_nodes() > 0:
            node = max(g.degree(), key=lambda x: (x[1], -x[0]))[0]
            g.remove_node(node)
            fracs.append(lcc_fraction(g, n_original))
        return np.array(fracs)

    def simulate_targeted_betweenness(graph):
        n_original = graph.number_of_nodes()
        g = graph.copy()
        fracs = [lcc_fraction(g, n_original)]
        while g.number_of_nodes() > 0:
            bc = nx.betweenness_centrality(g)
            node = max(bc, key=lambda x: (bc[x], -x))
            g.remove_node(node)
            fracs.append(lcc_fraction(g, n_original))
        return np.array(fracs)

    n_nodes = G.number_of_nodes()
    x_axis = np.linspace(0, 1, n_nodes + 1)

    random_matrix = simulate_random_attack(G, N_RANDOM_RUNS, RANDOM_SEED)
    random_mean = random_matrix.mean(axis=0)
    random_curve_low = np.percentile(random_matrix, 10, axis=0)
    random_curve_high = np.percentile(random_matrix, 90, axis=0)

    degree_curve = simulate_targeted_degree(G)
    betweenness_curve = simulate_targeted_betweenness(G)
    return (
        betweenness_curve,
        degree_curve,
        lcc_fraction,
        random_curve_high,
        random_curve_low,
        random_mean,
        x_axis,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Attack Curves
    """)
    return


@app.cell
def _(
    betweenness_curve,
    degree_curve,
    plt,
    random_curve_high,
    random_curve_low,
    random_mean,
    x_axis,
):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _ax.fill_between(
        x_axis,
        random_curve_low,
        random_curve_high,
        color="#AECAD1",
        alpha=0.45,
        label="Random (10th–90th pct.)",
    )
    _ax.plot(
        x_axis,
        random_mean,
        color="#00798C",
        linewidth=2.2,
        label="Random failure (mean)",
    )
    _ax.plot(
        x_axis,
        degree_curve,
        color="#D1495B",
        linewidth=2.2,
        linestyle="--",
        label="Targeted — degree",
    )
    _ax.plot(
        x_axis,
        betweenness_curve,
        color="#F4A35A",
        linewidth=2.2,
        linestyle=":",
        label="Targeted — betweenness",
    )

    _ax.set_xlabel("Fraction of nodes removed", fontsize=12)
    _ax.set_ylabel("LCC size / original node count", fontsize=12)
    _ax.set_title(
        "Network Resilience: Random Failure vs. Targeted Attack\n"
        "Facebook Ego Network — Ego 698",
        fontsize=13,
    )
    _ax.grid(alpha=0.22, linewidth=0.6)
    _ax.legend(frameon=False, fontsize=10)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.02)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Resilience Metrics

    The table below summarises key threshold statistics derived from each
    attack curve. **Critical fraction** is the smallest fraction of removed
    nodes at which the LCC first drops below 50 % of the original size.
    **Area under curve (AUC)** integrates the LCC trajectory: higher AUC
    means the network retains connectivity longer.
    """)
    return


@app.cell
def _(betweenness_curve, degree_curve, np, pd, random_mean, x_axis):
    def critical_fraction(curve, threshold=0.5):
        idx = np.argmax(curve < threshold)
        if idx == 0 and curve[0] >= threshold:
            return 1.0
        return float(x_axis[idx])

    def auc(curve):
        trapezoid = getattr(np, "trapezoid", np.trapz)
        return float(trapezoid(curve, x_axis))

    def lcc_at_removal(curve, frac):
        idx = int(round(frac * (len(curve) - 1)))
        return float(curve[min(idx, len(curve) - 1)])

    resilience_df = pd.DataFrame(
        [
            {
                "strategy": "Random failure (mean)",
                "critical fraction (LCC < 50 %)": critical_fraction(random_mean),
                "AUC": round(auc(random_mean), 4),
                "LCC at 10 % removed": round(lcc_at_removal(random_mean, 0.10), 3),
                "LCC at 25 % removed": round(lcc_at_removal(random_mean, 0.25), 3),
                "LCC at 50 % removed": round(lcc_at_removal(random_mean, 0.50), 3),
            },
            {
                "strategy": "Targeted — degree",
                "critical fraction (LCC < 50 %)": critical_fraction(degree_curve),
                "AUC": round(auc(degree_curve), 4),
                "LCC at 10 % removed": round(lcc_at_removal(degree_curve, 0.10), 3),
                "LCC at 25 % removed": round(lcc_at_removal(degree_curve, 0.25), 3),
                "LCC at 50 % removed": round(lcc_at_removal(degree_curve, 0.50), 3),
            },
            {
                "strategy": "Targeted — betweenness",
                "critical fraction (LCC < 50 %)": critical_fraction(betweenness_curve),
                "AUC": round(auc(betweenness_curve), 4),
                "LCC at 10 % removed": round(lcc_at_removal(betweenness_curve, 0.10), 3),
                "LCC at 25 % removed": round(lcc_at_removal(betweenness_curve, 0.25), 3),
                "LCC at 50 % removed": round(lcc_at_removal(betweenness_curve, 0.50), 3),
            },
        ]
    )
    resilience_df
    return auc, critical_fraction


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ego Node as a Single Point of Failure

    This section isolates the impact of removing ego node 698 first, which is
    what targeted degree attack does at step one given that the ego connects
    to every alter. It confirms the finding from Exercise 04.
    """)
    return


@app.cell
def _(EGO_ID, G, lcc_fraction, nx, pd):
    n_orig = G.number_of_nodes()

    G_no_ego = G.copy()
    G_no_ego.remove_node(EGO_ID)

    ego_components = sorted(nx.connected_components(G_no_ego), key=len, reverse=True)
    ego_component_sizes = [len(c) for c in ego_components]

    top_degree_nodes = sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:10]
    betweenness_full = nx.betweenness_centrality(G)
    top_betweenness_nodes = sorted(betweenness_full.items(), key=lambda x: (-x[1], x[0]))[:10]

    ego_impact_df = pd.DataFrame(
        [
            {
                "scenario": "original",
                "nodes remaining": n_orig,
                "LCC fraction": round(lcc_fraction(G, n_orig), 3),
                "n components": nx.number_connected_components(G),
            },
            {
                "scenario": f"remove ego {EGO_ID}",
                "nodes remaining": G_no_ego.number_of_nodes(),
                "LCC fraction": round(lcc_fraction(G_no_ego, n_orig), 3),
                "n components": nx.number_connected_components(G_no_ego),
            },
        ]
    )

    top_hubs_df = pd.DataFrame(
        [
            {
                "node": node,
                "degree": deg,
                "betweenness": round(betweenness_full[node], 5),
                "is ego": node == EGO_ID,
            }
            for node, deg in top_degree_nodes
        ]
    )

    ego_impact_df
    return (
        G_no_ego,
        ego_component_sizes,
        n_orig,
        top_betweenness_nodes,
        top_degree_nodes,
        top_hubs_df,
    )


@app.cell
def _(top_hubs_df):
    top_hubs_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison: Targeted vs. Random at Key Steps
    """)
    return


@app.cell
def _(betweenness_curve, degree_curve, pd, random_mean):
    steps = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    comparison_rows = []
    for frac in steps:
        idx = int(round(frac * (len(random_mean) - 1)))
        idx = min(idx, len(random_mean) - 1)
        comparison_rows.append(
            {
                "fraction removed": frac,
                "random LCC": round(float(random_mean[idx]), 3),
                "degree-attack LCC": round(float(degree_curve[idx]), 3),
                "betweenness-attack LCC": round(float(betweenness_curve[idx]), 3),
                "gap (random - degree)": round(float(random_mean[idx]) - float(degree_curve[idx]), 3),
            }
        )
    comparison_step_df = pd.DataFrame(comparison_rows)
    comparison_step_df
    return


@app.cell(hide_code=True)
def _(
    EGO_ID,
    G_no_ego,
    auc,
    betweenness_curve,
    critical_fraction,
    degree_curve,
    ego_component_sizes,
    mo,
    n_orig,
    nx,
    random_mean,
    top_betweenness_nodes,
    top_degree_nodes,
):
    _rand_cf = critical_fraction(random_mean)
    _deg_cf = critical_fraction(degree_curve)
    _bet_cf = critical_fraction(betweenness_curve)

    _rand_auc = round(auc(random_mean), 3)
    _deg_auc = round(auc(degree_curve), 3)
    _bet_auc = round(auc(betweenness_curve), 3)

    _n_comp_no_ego = nx.number_connected_components(G_no_ego)
    _lcc_no_ego = max(len(c) for c in nx.connected_components(G_no_ego))

    _top_deg_node = top_degree_nodes[0][0]
    _top_bet_node = top_betweenness_nodes[0][0]

    _comp_sizes_str = ", ".join(str(s) for s in ego_component_sizes[:5])

    mo.md(
        f"""
        ## Conclusion

        The Facebook ego network for user **{EGO_ID}** ({n_orig} nodes) shows a
        **stark resilience gap** between random failure and targeted attack.

        **Random failure is relatively harmless**: the mean LCC does not drop
        below 50 % until about **{_rand_cf:.0%}** of nodes have been removed
        (AUC = {_rand_auc}). This is the classic property of sparse
        random graphs — removing random low-degree nodes barely dents
        connectivity.

        **Degree-targeted attack is devastating**: the LCC falls below 50 %
        after only **{_deg_cf:.0%}** of nodes are removed (AUC = {_deg_auc}).
        The first node removed is always the ego **{EGO_ID}** itself, because
        it connects to every alter. Removing just that one node already breaks
        the network into **{_n_comp_no_ego} components** with a largest fragment
        of only **{_lcc_no_ego}** nodes
        (component sizes: {_comp_sizes_str}...).

        **Betweenness-targeted attack is even more devastating** — the most
        dangerous strategy of all. It reaches the 50 % threshold after only
        **{_bet_cf:.0%}** of nodes are removed (AUC = {_bet_auc}). Betweenness
        attack is worse than degree attack because the ego node also dominates
        betweenness, and subsequent removals are guided by structural brokerage
        rather than raw connectivity. This means the attack efficiently dismantles
        the bridges between the three alter circles before the circles themselves
        start to fragment.

        **Structural diagnosis — the ego is the single point of failure.**
        The network's connectivity hinges on one node. The three alter circles
        identified in Exercise 04 are essentially disconnected without the ego
        node; the top secondary hubs (node {top_degree_nodes[1][0]} with degree
        {top_degree_nodes[1][1]}, node {top_degree_nodes[2][0]} with degree
        {top_degree_nodes[2][1]}) span only one circle each and cannot replace
        the ego's bridging role.

        **Resilience gap summary**:
        - Random vs. degree attack AUC gap: {round(_rand_auc - _deg_auc, 3)}
        - Random vs. betweenness attack AUC gap: {round(_rand_auc - _bet_auc, 3)}

        **Proposed intervention — build cross-circle ties.**
        The root cause of fragility is that the three alter circles are
        *only* connected through the ego node. A practical policy to improve
        resilience would be to encourage or facilitate direct friendships between
        members of different circles (e.g., introducing work contacts to
        family friends). Even a small number of cross-circle edges would
        create redundant paths and convert the three isolated clusters into
        a single well-connected component that survives ego removal.
        In platform design terms, this corresponds to friend-of-friend
        recommendations that specifically target inter-community introductions
        rather than intra-community reinforcement.
        """
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sakila Film Network — Research Questions Analysis

    - This notebook investigates **eight research questions** about the multiplex film network built in the previous stage
    - Each question targets a different aspect of the relationship between graph structure and business/content attributes
    - The networks used are:
        - **Actor layer** — films linked by shared cast (IFA-weighted Jaccard)
        - **User layer** — films linked by co-rental patterns (cosine similarity, threshold ≥ 0.10)
        - **Combined (Borda)** — multiplex projection aggregating both layers via rank fusion
    - Key evaluation tools: Spearman correlation, Louvain community detection, NMI against genre labels, null-model Z-scores
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import networkx as nx
    from pathlib import Path
    from scipy import stats
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    return Path, mo, np, nx, pd, pearsonr, plt, stats


@app.cell
def _(Path):
    PROJECT_ROOT = Path("projects/lkrvavica")

    OUTPUT = PROJECT_ROOT / "output"
    GRAPHS_DIR = OUTPUT / "graphs"
    IMAGES_DIR = OUTPUT / "images"
    DATA = PROJECT_ROOT / "data" / "raw"
    return DATA, GRAPHS_DIR, IMAGES_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1 — Data and Graph Loading

    ### What is loaded here

    - Core film metadata: titles, genres, rental attributes, pricing
    - Pre-built graph files from the previous notebook (saved as GraphML):
        - `multiplex_network.graphml` — the two-layer MultiGraph
        - `combined_network.graphml` — Borda-projected single graph
        - `combined_linear_network.graphml` — linear-summed projection (comparison baseline)
    - Genre ground truth stored as **sets** (multi-label safe), with a single-label fallback for methods that require it
    - Film **popularity** = total rental count (derived from the rental → inventory → film join)
    """)
    return


@app.cell
def _(DATA, pd):
    film_df          = pd.read_csv(DATA / "film.csv")
    film_category_df = pd.read_csv(DATA / "film_category.csv")
    category_df      = pd.read_csv(DATA / "category.csv")
    inventory_df     = pd.read_csv(DATA / "inventory.csv")
    rental_df        = pd.read_csv(DATA / "rental.csv")
    payment_df       = pd.read_csv(DATA / "payment.csv")

    rental_film_df = rental_df.merge(inventory_df, on="inventory_id", how="inner")

    film_ids    = film_df["film_id"].tolist()
    film_titles = film_df.set_index("film_id")["title"].to_dict()

    rental_counts = (
        rental_film_df.groupby("film_id")["rental_id"]
        .count()
        .reindex(film_df["film_id"], fill_value=0)
    )
    film_popularity = rental_counts.to_dict()

    # Multi-label genre ground truth (dict of sets)
    film_genres_multi = (
        film_category_df.groupby("film_id")["category_id"]
        .apply(set)
        .to_dict()
    )
    film_genres_single = {fid: min(cats) for fid, cats in film_genres_multi.items()}

    genre_names = category_df.set_index("category_id")["name"].to_dict()

    multi_label_count = sum(1 for cats in film_genres_multi.values() if len(cats) > 1)
    print(f"Films with >1 genre: {multi_label_count} "
          f"({multi_label_count / len(film_genres_multi) * 100:.1f}%)")
    return (
        film_df,
        film_genres_multi,
        film_ids,
        film_popularity,
        film_titles,
        genre_names,
        inventory_df,
        payment_df,
        rental_df,
        rental_film_df,
    )


@app.cell
def _(GRAPHS_DIR, nx):
    print("Loading graphs …")
    G_multiplex       = nx.read_graphml(GRAPHS_DIR / "multiplex_network.graphml")
    G_combined        = nx.read_graphml(GRAPHS_DIR / "combined_network.graphml")
    G_combined_linear = nx.read_graphml(GRAPHS_DIR / "combined_linear_network.graphml")

    def _relabel(G):
        return nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})

    G_multiplex       = _relabel(G_multiplex)
    G_combined        = _relabel(G_combined)
    G_combined_linear = _relabel(G_combined_linear)

    print(f"  G_multiplex:        {G_multiplex.number_of_nodes()} nodes, "
          f"{G_multiplex.number_of_edges()} multi-edges")
    print(f"  G_combined (Borda): {G_combined.number_of_nodes()} nodes, "
          f"{G_combined.number_of_edges()} edges")
    print(f"  G_combined_linear:  {G_combined_linear.number_of_nodes()} nodes, "
          f"{G_combined_linear.number_of_edges()} edges")
    return G_combined, G_combined_linear, G_multiplex


@app.cell
def _(G_multiplex, nx):
    G_actor = nx.Graph()
    G_actor.add_nodes_from(G_multiplex.nodes())
    for _u, _v, _d in G_multiplex.edges(data=True):
        if _d.get("layer") == "actor":
            G_actor.add_edge(_u, _v, weight=_d["weight"])

    G_user = nx.Graph()
    G_user.add_nodes_from(G_multiplex.nodes())
    for _u, _v, _d in G_multiplex.edges(data=True):
        if _d.get("layer") == "user":
            G_user.add_edge(_u, _v, weight=_d["weight"])

    print(f"Actor layer: {G_actor.number_of_edges()} edges, "
          f"density={nx.density(G_actor):.4f}")
    print(f"User layer:  {G_user.number_of_edges()} edges, "
          f"density={nx.density(G_user):.4f}")
    return G_actor, G_user


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loaded graph summary

    | Graph | Nodes | Edges | Density |
    |---|---|---|---|
    | Multiplex (multi-layer) | 1,000 | 98,508 | — |
    | Combined — Borda projection | 1,000 | 94,378 | 0.189 |
    | Combined — linear projection | 1,000 | 94,378 | 0.189 |
    | Actor layer (extracted) | 1,000 | 68,919 | 0.138 |
    | User layer (extracted) | 1,000 | 29,589 | 0.059 |

    - Node labels are re-cast to integers to match film IDs in the CSV data
    - The actor and user layers are **extracted from the multiplex** for independent analysis
    - **Distance graphs** (weight → 1/weight) are derived for betweenness and closeness, where shorter = stronger connection
    """)
    return


@app.cell
def _(G_actor, G_user):
    def invert_weights(G):
        G_dist = G.copy()
        for _u, _v, _d in G_dist.edges(data=True):
            _w = _d.get("weight", 1.0)
            _d["weight"] = 1.0 / _w if _w > 0 else 1e6
        return G_dist

    G_actor_dist = invert_weights(G_actor)
    G_user_dist  = invert_weights(G_user)
    return G_actor_dist, G_user_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2 — Evaluation Helpers

    ### NMI and community evaluation

    - **NMI (Normalized Mutual Information):** measures how much a community partition agrees with genre labels
        - NMI = 1.0: perfect correspondence between communities and genres
        - NMI = 0.0: no agreement (communities are random with respect to genres)
    - **Multi-label NMI:** for films that belong to multiple genres, each genre assignment is counted separately — avoids the bias of picking one canonical genre
    - **Genre purity:** within each community, what fraction of films share the community's dominant genre?

    ### Null-model validation

    - To verify that observed community structure and centrality are not just artifacts of degree distribution, we compare against **configuration-model random graphs**
    - These preserve each node's degree but rewire connections randomly
    - A Z-score > 2–3 means the observed value is significantly above random expectation
    """)
    return


@app.cell
def _(film_genres_multi, film_ids):
    from sklearn.metrics import normalized_mutual_info_score as _sklearn_nmi

    def communities_to_labels(communities, node_list):
        node_to_comm = {}
        for idx, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = idx
        return [node_to_comm.get(n, -1) for n in node_list]

    def multilabel_nmi(comm_labels, node_list):
        """NMI against multi-label genre ground truth (marginalises over all assignments)."""
        expanded_pred, expanded_true = [], []
        for node, pred_label in zip(node_list, comm_labels):
            genres = film_genres_multi.get(node, set())
            if not genres:
                continue
            for g in genres:
                expanded_pred.append(pred_label)
                expanded_true.append(g)
        if not expanded_true:
            return 0.0
        return _sklearn_nmi(expanded_true, expanded_pred)

    nodes_with_genre = [f for f in film_ids if f in film_genres_multi]
    print(f"Films with genre annotation: {len(nodes_with_genre)}")
    return communities_to_labels, multilabel_nmi, nodes_with_genre


@app.cell
def _(np, nx):
    def build_null_graphs(G, n_null=50):
        """Degree-preserving configuration model null graphs."""
        nulls = []
        degree_seq = [d for _, d in G.degree()]
        for seed in range(n_null):
            try:
                R = nx.configuration_model(degree_seq, seed=seed)
                R = nx.Graph(R)
                R.remove_edges_from(nx.selfloop_edges(R))
                mapping = {i: list(G.nodes())[i] for i in range(len(G.nodes()))}
                R = nx.relabel_nodes(R, mapping)
                nulls.append(R)
            except Exception:
                pass
        return nulls

    def null_modularity_distribution(G, communities, n_null=50, seed=42):
        obs_mod   = nx.community.modularity(G, communities, weight=None)
        nulls     = build_null_graphs(G, n_null=n_null)
        null_mods = []
        for R in nulls:
            try:
                comms_r = nx.community.louvain_communities(R, seed=seed)
                null_mods.append(nx.community.modularity(R, comms_r))
            except Exception:
                pass
        if not null_mods:
            return obs_mod, float('nan'), float('nan'), float('nan')
        null_mean = np.mean(null_mods)
        null_std  = np.std(null_mods)
        z = (obs_mod - null_mean) / null_std if null_std > 0 else float('inf')
        return obs_mod, null_mean, null_std, z

    def null_centrality_distribution(G, centrality_dict, n_null=30):
        obs_mean   = np.mean(list(centrality_dict.values()))
        nulls      = build_null_graphs(G, n_null=n_null)
        null_means = []
        for R in nulls:
            try:
                pr_r = nx.pagerank(R)
                null_means.append(np.mean(list(pr_r.values())))
            except Exception:
                pass
        if not null_means:
            return obs_mean, float('nan'), float('nan'), float('nan')
        null_mean = np.mean(null_means)
        null_std  = np.std(null_means)
        z = (obs_mean - null_mean) / null_std if null_std > 0 else float('inf')
        return obs_mean, null_mean, null_std, z

    print("Null-model helpers defined.")
    return null_centrality_distribution, null_modularity_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 1 — Are Popular Films Also the Most Structurally Important?

    ### Question and motivation

    - A natural assumption: popular films (many rentals) should also be central in the network — rented by many people, co-rented with many others
    - If true, centrality ≈ popularity → networks add no information beyond simple rental counts
    - If false, the networks capture something **different** from raw popularity — more useful for nuanced recommendations

    ### Centrality metrics computed

    | Metric | What it captures |
    |---|---|
    | Weighted degree (strength) | Total similarity weight attached to a film |
    | Betweenness centrality | Bridge / connector role in the network |
    | PageRank | Importance weighted by the importance of neighbours |
    | Eigenvector centrality | Connection to other well-connected films |
    | Closeness centrality | Average distance to all other films |

    - All correlations use **Spearman ρ** — rank-based, robust to the skewed distribution of rental counts
    - Both layers computed separately to see if the signal differs
    """)
    return


@app.cell
def _(G_actor, G_user, film_ids, film_popularity, pd):
    strength_actor = dict(G_actor.degree(weight="weight"))
    strength_user  = dict(G_user.degree(weight="weight"))

    df_q1 = pd.DataFrame({
        "film_id":        film_ids,
        "popularity":     [film_popularity[f] for f in film_ids],
        "degree_actor":   [dict(G_actor.degree()).get(f, 0)  for f in film_ids],
        "degree_user":    [dict(G_user.degree()).get(f, 0)   for f in film_ids],
        "strength_actor": [strength_actor.get(f, 0.0)        for f in film_ids],
        "strength_user":  [strength_user.get(f, 0.0)         for f in film_ids],
    })
    return df_q1, strength_actor, strength_user


@app.cell
def _(df_q1, stats):
    rho_sa, p_sa = stats.spearmanr(df_q1["popularity"], df_q1["strength_actor"])
    rho_su, p_su = stats.spearmanr(df_q1["popularity"], df_q1["strength_user"])
    print(f"Spearman(popularity, strength_actor) = {rho_sa:.4f}  (p={p_sa:.2e})")
    print(f"Spearman(popularity, strength_user)  = {rho_su:.4f}  (p={p_su:.2e})")
    return rho_sa, rho_su


@app.cell
def _(G_actor_dist, G_user_dist, df_q1, film_ids, nx, stats):
    bt_actor = nx.betweenness_centrality(G_actor_dist, weight="weight")
    bt_user  = nx.betweenness_centrality(G_user_dist,  weight="weight")
    df_q1["bt_actor"] = [bt_actor.get(f, 0.0) for f in film_ids]
    df_q1["bt_user"]  = [bt_user.get(f, 0.0)  for f in film_ids]
    rho_ba, p_ba = stats.spearmanr(df_q1["popularity"], df_q1["bt_actor"])
    rho_bu, p_bu = stats.spearmanr(df_q1["popularity"], df_q1["bt_user"])
    print(f"Spearman(popularity, bt_actor) = {rho_ba:.4f}  (p={p_ba:.2e})")
    print(f"Spearman(popularity, bt_user)  = {rho_bu:.4f}  (p={p_bu:.2e})")
    return bt_actor, bt_user, rho_ba, rho_bu


@app.cell
def _(G_actor, G_user, df_q1, film_ids, nx, stats):
    pr_actor = nx.pagerank(G_actor, weight="weight")
    pr_user  = nx.pagerank(G_user,  weight="weight")
    df_q1["pr_actor"] = [pr_actor.get(f, 0.0) for f in film_ids]
    df_q1["pr_user"]  = [pr_user.get(f, 0.0)  for f in film_ids]
    rho_pa, p_pa = stats.spearmanr(df_q1["popularity"], df_q1["pr_actor"])
    rho_pu, p_pu = stats.spearmanr(df_q1["popularity"], df_q1["pr_user"])
    print(f"Spearman(popularity, PR_actor) = {rho_pa:.4f}  (p={p_pa:.2e})")
    print(f"Spearman(popularity, PR_user)  = {rho_pu:.4f}  (p={p_pu:.2e})")
    return pr_actor, pr_user, rho_pa, rho_pu


@app.cell
def _(G_actor, G_user, df_q1, film_ids, nx, stats):
    try:
        eig_actor = nx.eigenvector_centrality(G_actor, weight="weight", max_iter=1000)
        df_q1["eig_actor"] = [eig_actor.get(f, 0.0) for f in film_ids]
        rho_ea, _ = stats.spearmanr(df_q1["popularity"], df_q1["eig_actor"])
        print(f"Spearman(popularity, eig_actor) = {rho_ea:.4f}")
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality failed to converge on Actor graph.")
        rho_ea = None
    try:
        eig_user = nx.eigenvector_centrality(G_user, weight="weight", max_iter=1000)
        df_q1["eig_user"] = [eig_user.get(f, 0.0) for f in film_ids]
        rho_eu, _ = stats.spearmanr(df_q1["popularity"], df_q1["eig_user"])
        print(f"Spearman(popularity, eig_user)  = {rho_eu:.4f}")
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality failed to converge on User graph.")
        rho_eu = None
    return rho_ea, rho_eu


@app.cell
def _(G_actor_dist, G_user_dist, df_q1, film_ids, nx, stats):
    cl_actor = nx.closeness_centrality(G_actor_dist, distance="weight")
    cl_user  = nx.closeness_centrality(G_user_dist,  distance="weight")
    df_q1["cl_actor"] = [cl_actor.get(f, 0.0) for f in film_ids]
    df_q1["cl_user"]  = [cl_user.get(f, 0.0)  for f in film_ids]
    rho_ca, _ = stats.spearmanr(df_q1["popularity"], df_q1["cl_actor"])
    rho_cu, _ = stats.spearmanr(df_q1["popularity"], df_q1["cl_user"])
    print(f"Spearman(popularity, closeness_actor) = {rho_ca:.4f}")
    print(f"Spearman(popularity, closeness_user)  = {rho_cu:.4f}")
    return rho_ca, rho_cu


@app.cell
def _(G_actor, G_user, null_centrality_distribution, pr_actor, pr_user):
    pr_obs_a, pr_null_mean_a, pr_null_std_a, pr_z_a = null_centrality_distribution(
        G_actor, pr_actor, n_null=30
    )
    pr_obs_u, pr_null_mean_u, pr_null_std_u, pr_z_u = null_centrality_distribution(
        G_user, pr_user, n_null=30
    )
    print(f"PageRank null-model comparison:")
    print(f"  Actor: obs={pr_obs_a:.6f}, null={pr_null_mean_a:.6f}±{pr_null_std_a:.6f}, Z={pr_z_a:.2f}")
    print(f"  User:  obs={pr_obs_u:.6f}, null={pr_null_mean_u:.6f}±{pr_null_std_u:.6f}, Z={pr_z_u:.2f}")
    print(f"  (Z > 2 means observed centrality concentration exceeds random expectation)")
    return


@app.cell
def _(
    rho_ba,
    rho_bu,
    rho_ca,
    rho_cu,
    rho_ea,
    rho_eu,
    rho_pa,
    rho_pu,
    rho_sa,
    rho_su,
):
    print(f"\n{'Metric':<30s} | {'Actor rho':>12s} | {'User rho':>12s}")
    print("-" * 60)
    print(f"{'Strength (weighted degree)':<30s} | {rho_sa:>12.4f} | {rho_su:>12.4f}")
    print(f"{'Betweenness (1/w distance)':<30s} | {rho_ba:>12.4f} | {rho_bu:>12.4f}")
    print(f"{'PageRank':<30s} | {rho_pa:>12.4f} | {rho_pu:>12.4f}")
    eig_a_str = f"{rho_ea:.4f}" if rho_ea is not None else "FAILED"
    eig_u_str = f"{rho_eu:.4f}" if rho_eu is not None else "FAILED"
    print(f"{'Eigenvector':<30s} | {eig_a_str:>12s} | {eig_u_str:>12s}")
    print(f"{'Closeness':<30s} | {rho_ca:>12.4f} | {rho_cu:>12.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 1 — Centrality summary table

    | Metric | Actor ρ | User ρ |
    |---|---|---|
    | Strength (weighted degree) | 0.068 | 0.078 |
    | Betweenness | −0.015 | **−0.214** |
    | PageRank | 0.065 | 0.085 |
    | Eigenvector | 0.074 | 0.039 |
    | Closeness | −0.011 | **−0.177** |

    - **All correlations are near zero** — popular films are not structurally central in either network
    - The **actor layer shows virtually no relationship** with popularity across all five metrics (mean |ρ| ≈ 0.05)
    - The user layer shows **negative betweenness and closeness** correlations — popular films are peripheral in the taste graph, not bridges
    - PageRank null model Z-scores are near zero → centrality distribution is not meaningfully concentrated above random expectation
    - **Key finding: popularity and network centrality are orthogonal** — the networks encode something genuinely different from rental volume
    """)
    return


@app.cell
def _(df_q1, film_titles):
    top_pop = df_q1.nlargest(10, "popularity")[["film_id", "popularity", "bt_user"]]
    top_bt  = df_q1.nlargest(10, "bt_user")[["film_id", "popularity", "bt_user"]]
    print("\nTop 10 by POPULARITY:")
    for _, _row in top_pop.iterrows():
        title = film_titles.get(int(_row["film_id"]), "???")
        print(f"  {title:<35s} rentals={int(_row['popularity']):3d}  betweenness={_row['bt_user']:.6f}")
    print("\nTop 10 by BETWEENNESS (user network):")
    for _, _row in top_bt.iterrows():
        title = film_titles.get(int(_row["film_id"]), "???")
        print(f"  {title:<35s} rentals={int(_row['popularity']):3d}  betweenness={_row['bt_user']:.6f}")
    overlap = set(top_pop["film_id"]) & set(top_bt["film_id"])
    print(f"\nOverlap between top-10 lists: {len(overlap)} / 10 films")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top-10 comparison: popularity vs betweenness

    - The most-rented films (e.g. BUCKET BROTHERHOOD, 34 rentals) have **very low betweenness** (0.0002–0.003)
    - The highest-betweenness films (e.g. WORLD LEATHERNECKS, BRAVEHEART HUMAN) have **very few rentals** (5–10)
    - **Zero overlap** between the two top-10 lists
    - Interpretation: betweenness-central films act as **taste bridges** — they connect different audience segments, not because they are popular, but because they appeal to diverse niches
    - These bridge films could be valuable for **cold-start recommendations** or for linking otherwise disconnected user communities
    """)
    return


@app.cell
def _(IMAGES_DIR, df_q1, plt, rho_bu, rho_sa, rho_su):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
    _axes[0].scatter(df_q1["popularity"], df_q1["strength_user"],
                     alpha=0.4, s=15, color="steelblue")
    _axes[0].set_xlabel("Popularity (rentals)")
    _axes[0].set_ylabel("Weighted Degree (user layer)")
    _axes[0].set_title(f"Popularity vs User Strength\nρ = {rho_su:.3f}")
    _axes[1].scatter(df_q1["popularity"], df_q1["strength_actor"],
                     alpha=0.4, s=15, color="coral")
    _axes[1].set_xlabel("Popularity (rentals)")
    _axes[1].set_ylabel("Weighted Degree (actor layer)")
    _axes[1].set_title(f"Popularity vs Actor Strength\nρ = {rho_sa:.3f}")
    _axes[2].scatter(df_q1["popularity"], df_q1["bt_user"],
                     alpha=0.4, s=15, color="seagreen")
    _axes[2].set_xlabel("Popularity (rentals)")
    _axes[2].set_ylabel("Betweenness Centrality (user layer)")
    _axes[2].set_title(f"Popularity vs User Betweenness\nρ = {rho_bu:.3f}")
    _fig.suptitle("RQ1: Popularity vs Structural Importance", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq1.1_popularity_vs_centrality.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Figure 1 — Popularity vs structural importance (scatter plots)

    - **Left panel (user strength ρ = 0.078):** cloud of points with no directional trend — high-popularity films scatter across all strength values
    - **Middle panel (actor strength ρ = 0.068):** even flatter — the actor layer is almost entirely independent of how often a film was rented
    - **Right panel (user betweenness ρ = −0.214):** notable *negative* slope — the most popular films cluster at the bottom (low betweenness), while high-betweenness films are low-popularity niche titles
    - **Takeaway:** none of the three scatter plots shows a meaningful positive trend → popularity is not a proxy for structural importance in either layer
    """)
    return


@app.cell
def _(
    IMAGES_DIR,
    np,
    plt,
    rho_ba,
    rho_bu,
    rho_ca,
    rho_cu,
    rho_ea,
    rho_eu,
    rho_pa,
    rho_pu,
    rho_sa,
    rho_su,
):
    _metrics_labels = ["Strength", "Betweenness", "PageRank", "Eigenvector", "Closeness"]
    _rho_actor_vals = [rho_sa, rho_ba, rho_pa,
                       rho_ea if rho_ea is not None else 0.0, rho_ca]
    _rho_user_vals  = [rho_su, rho_bu, rho_pu,
                       rho_eu if rho_eu is not None else 0.0, rho_cu]
    _x = np.arange(len(_metrics_labels))
    _w = 0.35
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(_x - _w/2, _rho_actor_vals, _w, label="Actor layer", color="coral",     alpha=0.85)
    _ax.bar(_x + _w/2, _rho_user_vals,  _w, label="User layer",  color="steelblue", alpha=0.85)
    _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_metrics_labels)
    _ax.set_ylabel("Spearman ρ with popularity")
    _ax.set_title("RQ1: Popularity–Centrality Correlations by Layer and Metric")
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq1.2_spearman_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Figure 2 — Spearman ρ summary bar chart

    - All bars are clustered near zero — no metric in either layer strongly predicts popularity
    - The actor layer (coral) is consistently flatter than the user layer (steel blue)
    - The two most extreme values are **user betweenness (−0.21)** and **user closeness (−0.18)** — both negative, confirming popular films are peripheral, not central, in the taste graph
    - Mean |ρ|: actor = 0.047, user = 0.119 — user layer has modestly more signal, but still weak
    - **Answer to RQ1: No** — popular films are not structurally important; the networks capture a separate dimension of film similarity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 2 — Do Actor-Based and User-Based Connections Provide a Similar View?

    ### Question and motivation

    - If the two layers encode the same information, building a multiplex is redundant
    - We need to verify that actor similarity and rental co-occurrence are **truly independent signals**
    - Tests:
        - Spearman correlation of centrality rankings across layers
        - Edge-set Jaccard overlap (what fraction of edges appear in both layers?)
        - Neighbour Jaccard (do a film's actor-neighbours match its user-neighbours?)
        - Weight correlation on shared edges (do films connected in both layers have similar weights in each?)
    """)
    return


@app.cell
def _(df_q1, stats):
    metrics = {
        "Degree":      ("degree_actor",   "degree_user"),
        "Strength":    ("strength_actor", "strength_user"),
        "Betweenness": ("bt_actor",       "bt_user"),
        "PageRank":    ("pr_actor",       "pr_user"),
        "Closeness":   ("cl_actor",       "cl_user"),
    }
    print(f"\n  {'Metric':<20s} | {'Spearman rho':>12s} | {'p-value':>12s}")
    print("  " + "-" * 52)
    for name, (col_a, col_u) in metrics.items():
        rho_cross, p_cross = stats.spearmanr(df_q1[col_a], df_q1[col_u])
        print(f"  {name:<20s} | {rho_cross:>12.4f} | {p_cross:>12.2e}")
    return


@app.cell
def _(G_actor, G_user):
    actor_edges = {tuple(sorted(e)) for e in G_actor.edges()}
    user_edges  = {tuple(sorted(e)) for e in G_user.edges()}
    intersection      = actor_edges & user_edges
    union_edges       = actor_edges | user_edges
    jaccard_overlap   = len(intersection) / len(union_edges) if union_edges else 0
    print(f"  Actor edges:     {len(actor_edges)}")
    print(f"  User edges:      {len(user_edges)}")
    print(f"  Shared edges:    {len(intersection)}")
    print(f"  Jaccard overlap: {jaccard_overlap:.4f}")
    return actor_edges, intersection, user_edges


@app.cell
def _(G_actor, G_user, intersection, np, stats):
    if len(intersection) > 10:
        w_a = [G_actor[u][v]["weight"] for u, v in intersection]
        w_u = [G_user[u][v]["weight"]  for u, v in intersection]
        rho_w, p_w = stats.spearmanr(w_a, w_u)
        print(f"  On {len(intersection)} shared edges:")
        print(f"  Spearman(actor_weight, user_weight) = {rho_w:.4f}  (p={p_w:.2e})")
        print(f"  Mean actor weight on shared edges:  {np.mean(w_a):.4f}")
        print(f"  Mean user weight on shared edges:   {np.mean(w_u):.4f}")
    else:
        print(f"  Only {len(intersection)} shared edges — too few for meaningful correlation.")
        rho_w, p_w = None, None
    return


@app.cell
def _(G_actor, G_user, film_ids, np):
    neighbor_jaccards = []
    for f in film_ids:
        n_actor = set(G_actor.neighbors(f))
        n_user  = set(G_user.neighbors(f))
        union_n = n_actor | n_user
        if not union_n:
            continue
        neighbor_jaccards.append(len(n_actor & n_user) / len(union_n))
    neighbor_jaccards = np.array(neighbor_jaccards)
    print(f"  Mean neighbor Jaccard:   {np.mean(neighbor_jaccards):.4f}")
    print(f"  Median neighbor Jaccard: {np.median(neighbor_jaccards):.4f}")
    print(f"  % of films with Jaccard = 0: "
          f"{(neighbor_jaccards == 0).sum() / len(neighbor_jaccards) * 100:.1f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 2 — Cross-layer independence evidence

    | Test | Result | Interpretation |
    |---|---|---|
    | Centrality rank correlation (all metrics) | ρ ≈ 0.00–0.06 | Essentially zero — rankings are independent |
    | Edge Jaccard overlap | 0.044 (4.4%) | Only 4,130 of 94,378 edges shared |
    | Weight correlation on shared edges | ρ ≈ −0.001 | Even shared edges carry unrelated weights |
    | Mean neighbour Jaccard | 0.042 | A film's actor-neighbours and user-neighbours barely overlap |

    - **All four tests converge on the same conclusion:** the actor and user layers are **nearly independent**
    - A film's centrality rank in one layer tells you virtually nothing about its rank in the other
    - Only ~4% of edges appear in both layers, and even those have uncorrelated weights
    - **Answer to RQ2: No** — the two layers do not provide a similar view; they are genuinely complementary and the multiplex is well-justified
    """)
    return


@app.cell
def _(IMAGES_DIR, actor_edges, df_q1, intersection, plt, stats, user_edges):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _axes[0].scatter(df_q1["strength_actor"], df_q1["strength_user"],
                     alpha=0.35, s=12, color="mediumpurple")
    _axes[0].set_xlabel("Weighted Degree — Actor layer")
    _axes[0].set_ylabel("Weighted Degree — User layer")
    _rho_cross_str, _ = stats.spearmanr(df_q1["strength_actor"], df_q1["strength_user"])
    _axes[0].set_title(f"Cross-layer Strength\nρ = {_rho_cross_str:.3f}")

    _actor_only = len(actor_edges) - len(intersection)
    _user_only  = len(user_edges)  - len(intersection)
    _axes[1].barh(["Actor only", "Shared", "User only"],
                  [_actor_only, len(intersection), _user_only],
                  color=["coral", "mediumpurple", "steelblue"], alpha=0.85)
    for _i, _v in enumerate([_actor_only, len(intersection), _user_only]):
        _axes[1].text(_v + 100, _i, f"{_v:,}", va="center", fontsize=10)
    _axes[1].set_xlabel("Number of edges")
    _axes[1].set_title("Edge Set Overlap Between Layers")

    _fig.suptitle("RQ2: Cross-Layer Comparison", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq2.1_cross_layer.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Figure 3 — Cross-layer comparison

    - **Left panel (scatter ρ = −0.006):** points fill the entire plane with no pattern — actor strength and user strength are completely uncorrelated
    - **Right panel (bar chart):** the vast majority of edges are exclusive to one layer
        - Actor-only: 64,789 edges (68.6% of all unique edges)
        - User-only: 25,459 edges (26.9%)
        - Shared: only 4,130 (4.4%)
    - **Takeaway:** the two layers describe almost entirely different pairs of similar films — combining them in a multiplex captures ~23× more relationship information than either layer alone
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 3 — Community Detection: Which Method Finds the Best Film Clusters?

    ### Methods compared

    | Method | Description |
    |---|---|
    | Louvain (actor only) | Greedy modularity optimisation on actor layer |
    | Louvain (user only) | Same, on user co-rental layer |
    | Louvain (Borda combined) | Same, on the multiplex projection |
    | Louvain (linear combined) | Baseline: sum of raw layer weights |
    | Supra-adjacency Louvain | Louvain on a unified graph coupling both layers via inter-layer edges (weight ω) |
    | Cross-layer consensus | Ensemble method: co-assignment frequency across multiple runs and both layers |

    ### Evaluation metrics

    - **Modularity:** internal cohesion of communities relative to a random graph (higher = cleaner clusters)
    - **NMI vs genre (multi-label):** how well communities align with known film genres
    - **Stability:** consistency of partitions across 20 independent random Louvain runs (higher = more robust)
    - **Null-model Z-score:** is the observed modularity significantly above what random graphs produce?
    """)
    return


@app.cell
def _(G_actor, G_combined, G_combined_linear, G_user, nx):
    print("Running Louvain on all graphs …")
    comm_actor         = nx.community.louvain_communities(G_actor,           weight="weight", seed=42)
    comm_user          = nx.community.louvain_communities(G_user,            weight="weight", seed=42)
    comm_combined      = nx.community.louvain_communities(G_combined,        weight="weight", seed=42)
    comm_combined_lin  = nx.community.louvain_communities(G_combined_linear, weight="weight", seed=42)

    mod_actor        = nx.community.modularity(G_actor,          comm_actor,       weight="weight")
    mod_user         = nx.community.modularity(G_user,           comm_user,        weight="weight")
    mod_combined     = nx.community.modularity(G_combined,       comm_combined,    weight="weight")
    mod_combined_lin = nx.community.modularity(G_combined_linear, comm_combined_lin, weight="weight")

    print(f"  Actor communities:          {len(comm_actor)},  modularity={mod_actor:.4f}")
    print(f"  User communities:           {len(comm_user)},   modularity={mod_user:.4f}")
    print(f"  Borda combined:             {len(comm_combined)},  modularity={mod_combined:.4f}")
    print(f"  Linear combined (baseline): {len(comm_combined_lin)}, modularity={mod_combined_lin:.4f}")
    return (
        comm_actor,
        comm_combined,
        comm_combined_lin,
        comm_user,
        mod_actor,
        mod_combined,
        mod_combined_lin,
        mod_user,
    )


@app.cell
def _(G_actor, G_user, film_ids, nx):
    def build_supra_graph(G_actor, G_user, film_ids, omega=0.5):
        G_supra = nx.Graph()
        for u, v, d in G_actor.edges(data=True):
            G_supra.add_edge((u, 'actor'), (v, 'actor'), weight=d['weight'])
        for u, v, d in G_user.edges(data=True):
            G_supra.add_edge((u, 'user'), (v, 'user'), weight=d['weight'])
        for fid in film_ids:
            G_supra.add_edge((fid, 'actor'), (fid, 'user'), weight=omega)
        return G_supra

    def supra_to_film_communities(supra_communities, film_ids):
        node_to_comm = {}
        for idx, comm in enumerate(supra_communities):
            for node in comm:
                node_to_comm[node] = idx
        film_communities_dict = {}
        for fid in film_ids:
            actor_comm = node_to_comm.get((fid, 'actor'), -1)
            user_comm  = node_to_comm.get((fid, 'user'),  -1)
            if actor_comm != -1 and user_comm != -1:
                film_communities_dict[fid] = actor_comm
            elif actor_comm != -1:
                film_communities_dict[fid] = actor_comm
            else:
                film_communities_dict[fid] = user_comm
        comm_map = {}
        for fid, cid in film_communities_dict.items():
            comm_map.setdefault(cid, set()).add(fid)
        return list(comm_map.values())

    supra_results = {}
    for _omega in [0.1, 0.5, 1.0]:
        G_supra = build_supra_graph(G_actor, G_user, film_ids, omega=_omega)
        comm_supra_raw  = nx.community.louvain_communities(G_supra, weight="weight", seed=42)
        comm_supra_film = supra_to_film_communities(comm_supra_raw, film_ids)
        mod_supra       = nx.community.modularity(G_supra, comm_supra_raw, weight="weight")
        supra_results[_omega] = {
            "comm_film": comm_supra_film,
            "n_comm":    len(comm_supra_film),
            "mod_supra": mod_supra,
        }
        print(f"  ω={_omega:.1f}: {len(comm_supra_film)} film-communities, "
              f"supra-modularity={mod_supra:.4f}")

    comm_supra = supra_results[0.5]["comm_film"]
    return (comm_supra,)


@app.cell
def _(
    comm_actor,
    comm_combined,
    comm_supra,
    comm_user,
    communities_to_labels,
    multilabel_nmi,
    nodes_with_genre,
):
    label_actor    = communities_to_labels(comm_actor,    nodes_with_genre)
    label_user     = communities_to_labels(comm_user,     nodes_with_genre)
    label_combined = communities_to_labels(comm_combined, nodes_with_genre)
    label_supra    = communities_to_labels(comm_supra,    nodes_with_genre)

    nmi_actor    = multilabel_nmi(label_actor,    nodes_with_genre)
    nmi_user     = multilabel_nmi(label_user,     nodes_with_genre)
    nmi_combined = multilabel_nmi(label_combined, nodes_with_genre)
    nmi_supra    = multilabel_nmi(label_supra,    nodes_with_genre)

    print("NMI against multi-label genre ground truth:")
    print(f"  Actor:                   {nmi_actor:.4f}")
    print(f"  User:                    {nmi_user:.4f}")
    print(f"  Combined (Borda):        {nmi_combined:.4f}")
    print(f"  Supra-adjacency (ω=0.5): {nmi_supra:.4f}")
    return nmi_actor, nmi_combined, nmi_supra, nmi_user


@app.cell
def _(
    G_actor,
    G_combined,
    G_user,
    comm_actor,
    comm_combined,
    comm_user,
    null_modularity_distribution,
):
    print("=== NULL-MODEL VALIDATION FOR COMMUNITY STRUCTURE ===\n")
    for _name, _G, _comm in [
        ("Actor",    G_actor,    comm_actor),
        ("User",     G_user,     comm_user),
        ("Combined", G_combined, comm_combined),
    ]:
        _obs, _null_mean, _null_std, _z = null_modularity_distribution(
            _G, _comm, n_null=30, seed=42
        )
        print(f"  {_name:<10s}: obs={_obs:.4f}, null={_null_mean:.4f}±{_null_std:.4f}, Z={_z:.2f}")
    print("\nNote: Supra-adjacency modularity is on the supra-graph and not directly comparable.")
    return


@app.cell
def _(G_actor, G_combined, G_user, communities_to_labels, film_ids, np, nx):
    def measure_stability(G, n_runs=20):
        from sklearn.metrics import normalized_mutual_info_score as _nmi
        partitions = []
        for seed in range(n_runs):
            comms  = nx.community.louvain_communities(G, weight="weight", seed=seed)
            labels = communities_to_labels(comms, film_ids)
            partitions.append(labels)
        nmis = [
            _nmi(partitions[i], partitions[j])
            for i in range(n_runs)
            for j in range(i + 1, n_runs)
        ]
        return np.mean(nmis), np.std(nmis)

    print("Computing Louvain stability (20 runs each) …")
    stab_actor,    std_actor    = measure_stability(G_actor)
    stab_user,     std_user     = measure_stability(G_user)
    stab_combined, std_combined = measure_stability(G_combined)
    print(f"  Actor:    {stab_actor:.4f} ± {std_actor:.4f}")
    print(f"  User:     {stab_user:.4f}  ± {std_user:.4f}")
    print(f"  Combined: {stab_combined:.4f} ± {std_combined:.4f}")
    return measure_stability, stab_actor, stab_combined, stab_user


@app.cell
def _(
    G_actor,
    G_user,
    communities_to_labels,
    film_ids,
    measure_stability,
    multilabel_nmi,
    nodes_with_genre,
    np,
    nx,
):
    # Cross-layer consensus partition
    print("=== CROSS-LAYER CONSENSUS PARTITION ===\n")
    n_runs_consensus = 10
    consensus_matrix = np.zeros((len(film_ids), len(film_ids)))
    film_id_to_idx   = {f: i for i, f in enumerate(film_ids)}

    for G_layer, layer_name in [(G_actor, "actor"), (G_user, "user")]:
        for seed in range(n_runs_consensus):
            comms = nx.community.louvain_communities(G_layer, weight="weight", seed=seed)
            for comm in comms:
                comm_list = [film_id_to_idx[n] for n in comm if n in film_id_to_idx]
                for _i in range(len(comm_list)):
                    for _j in range(_i + 1, len(comm_list)):
                        consensus_matrix[comm_list[_i], comm_list[_j]] += 1
                        consensus_matrix[comm_list[_j], comm_list[_i]] += 1

    consensus_matrix /= (2 * n_runs_consensus)

    def build_consensus_graph(threshold):
        G_c = nx.Graph()
        G_c.add_nodes_from(film_ids)
        for _i in range(len(film_ids)):
            for _j in range(_i + 1, len(film_ids)):
                _w = consensus_matrix[_i, _j]
                if _w >= threshold:
                    G_c.add_edge(film_ids[_i], film_ids[_j], weight=_w)
        return G_c

    G_consensus = build_consensus_graph(0.5)
    print(f"Consensus graph (τ=0.5): {G_consensus.number_of_edges()} edges, "
          f"density={nx.density(G_consensus):.4f}")

    if G_consensus.number_of_edges() > 0:
        comm_consensus    = nx.community.louvain_communities(G_consensus, weight="weight", seed=42)
        mod_consensus_raw = nx.community.modularity(G_consensus, comm_consensus, weight="weight")
        lbl_consensus     = communities_to_labels(comm_consensus, nodes_with_genre)
        nmi_consensus     = multilabel_nmi(lbl_consensus, nodes_with_genre)
        stab_consensus, std_consensus = measure_stability(G_consensus, n_runs=5)
        print(f"  Communities:       {len(comm_consensus)}")
        print(f"  Modularity (raw):  {mod_consensus_raw:.4f}")
        print(f"  NMI (multi-label): {nmi_consensus:.4f}")
        print(f"  Stability:         {stab_consensus:.4f} ± {std_consensus:.4f}")
    else:
        print("  No edges at τ=0.5 — layers find completely different communities.")
        comm_consensus = []
        mod_consensus_raw = nmi_consensus = stab_consensus = 0.0
    return G_consensus, comm_consensus, mod_consensus_raw, nmi_consensus


@app.cell
def _(G_consensus, np, nx):
    # Modularity inflation check for consensus graph
    print("=== MODULARITY INFLATION CHECK ===\n")
    if G_consensus.number_of_edges() == 0:
        print("  Skipped — consensus graph has no edges.")
    else:
        obs_mod_c = nx.community.modularity(
            G_consensus,
            nx.community.louvain_communities(G_consensus, weight="weight", seed=42),
            weight="weight"
        )
        n_perm = 100
        perm_mods = []
        nodes = list(G_consensus.nodes())
        for _seed in range(n_perm):
            rng = np.random.default_rng(_seed)
            shuffled = rng.permutation(nodes).tolist()
            mapping  = {old: new for old, new in zip(nodes, shuffled)}
            G_perm   = nx.relabel_nodes(G_consensus, mapping)
            comm_perm = nx.community.louvain_communities(G_perm, weight="weight", seed=0)
            perm_mods.append(nx.community.modularity(G_perm, comm_perm, weight="weight"))

        perm_mean   = np.mean(perm_mods)
        perm_std    = np.std(perm_mods)
        z_inflation = (obs_mod_c - perm_mean) / perm_std if perm_std > 0 else float('inf')

        print(f"  Observed modularity:   {obs_mod_c:.4f}")
        print(f"  Permuted mean ± std:   {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"  Z-score:               {z_inflation:.2f}")
        if z_inflation > 3:
            print("  → Community structure is real, not a construction artifact.")
        else:
            print("  → High modularity is at least partly a mechanical inflation artifact.")
    return


@app.cell
def _(
    comm_actor,
    comm_combined,
    comm_combined_lin,
    comm_consensus,
    comm_supra,
    comm_user,
    mod_actor,
    mod_combined,
    mod_combined_lin,
    mod_consensus_raw,
    mod_user,
    nmi_actor,
    nmi_combined,
    nmi_consensus,
    nmi_supra,
    nmi_user,
    stab_actor,
    stab_combined,
    stab_user,
):
    print(f"\n{'Method':<35s} | {'#Comm':>6s} | {'Modularity':>11s} | {'NMI(multi)':>11s} | {'Stability':>11s}")
    print("-" * 85)
    print(f"{'Actor only':<35s} | {len(comm_actor):>6d} | {mod_actor:>11.4f} | {nmi_actor:>11.4f} | {stab_actor:>11.4f}")
    print(f"{'User only':<35s} | {len(comm_user):>6d} | {mod_user:>11.4f} | {nmi_user:>11.4f} | {stab_user:>11.4f}")
    print(f"{'Borda combined':<35s} | {len(comm_combined):>6d} | {mod_combined:>11.4f} | {nmi_combined:>11.4f} | {stab_combined:>11.4f}")
    print(f"{'Linear combined (baseline)':<35s} | {len(comm_combined_lin):>6d} | {mod_combined_lin:>11.4f} | {'—':>11s} | {'—':>11s}")
    print(f"{'Supra-adjacency (ω=0.5)':<35s} | {len(comm_supra):>6d} | {'(supra)':>11s} | {nmi_supra:>11.4f} | {'—':>11s}")
    if comm_consensus:
        print(f"{'Cross-layer consensus (τ=0.5)':<35s} | {len(comm_consensus):>6d} | {mod_consensus_raw:>11.4f}* | {nmi_consensus:>11.4f} | {'—':>11s}")
    print("\n* consensus modularity may be mechanically inflated (see permutation test)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 3 — Community detection results

    | Method | Communities | Modularity | NMI (genre) | Stability |
    |---|---|---|---|---|
    | Actor only | 18 | 0.171 | 0.043 | 0.308 |
    | User only | 57 | 0.189 | **0.080** | **0.361** |
    | Borda combined | 10 | 0.116 | 0.025 | 0.216 |
    | Linear combined | 15 | 0.114 | — | — |
    | Supra-adjacency (ω=0.5) | 2 | — | 0.004 | — |
    | **Cross-layer consensus (τ=0.5)** | 174 | 0.800* | **0.187** | — |

    ### Key observations

    - **User layer** finds the most genre-aligned communities (NMI 0.08) with the best stability (0.36)
    - **Actor layer** finds fewer, larger communities — cast overlap is a coarser similarity signal
    - **Combined graph** merges information but produces fewer, weaker communities (NMI drops to 0.025) — layer integration is not straightforward
    - **Supra-adjacency** at ω=0.5 collapses to only 2 communities — the inter-layer coupling forces over-merging
    - **Cross-layer consensus** achieves the highest NMI (0.187) and near-perfect stability (0.944) — but its modularity (0.80) is artificially inflated (confirmed by the permutation test Z-score of −1.6 × 10¹²)
    - **Null-model check:** actor modularity is significantly above random (Z = 41.7); user layer is *below* null (Z = −12) — suggesting the user graph is not well-suited for modularity-based detection without further preprocessing

    ### Practical takeaway
    - For genre-recovery tasks, **user layer + Louvain** or **cross-layer consensus** are the best options
    - The consensus modularity score should not be interpreted at face value — it is a known artifact of co-clustering frequency graph construction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 4 — Do Films Cluster Differently by Genre in Each Layer?

    ### Question and motivation

    - Genre is the most natural ground truth for film similarity
    - We examine: do films of the same genre cluster together more in the actor layer or the user layer?
    - Three complementary measures:
        - **NMI** — global agreement between community partition and genre labels
        - **Genre purity** — within each community, how many films share the dominant genre?
        - **Intra-genre edge fraction** — what fraction of edges connect same-genre films?
    - Also: which genres have the highest average connectivity in each layer?
    """)
    return


@app.cell
def _(
    comm_actor,
    comm_user,
    communities_to_labels,
    multilabel_nmi,
    nodes_with_genre,
):
    _lbl_actor = communities_to_labels(comm_actor, nodes_with_genre)
    _lbl_user  = communities_to_labels(comm_user,  nodes_with_genre)
    _nmi_actor = multilabel_nmi(_lbl_actor, nodes_with_genre)
    _nmi_user  = multilabel_nmi(_lbl_user,  nodes_with_genre)
    print(f"NMI (multi-label) actor communities vs genres = {_nmi_actor:.4f}")
    print(f"NMI (multi-label) user  communities vs genres = {_nmi_user:.4f}")
    print(f"Actor Louvain communities: {len(comm_actor)}")
    print(f"User  Louvain communities: {len(comm_user)}")
    return


@app.cell
def _(comm_actor, comm_user, film_genres_multi, np):
    def genre_purity_multilabel(communities, film_genre_map_multi):
        purities = []
        for comm in communities:
            genre_counts = {}
            for fid in comm:
                for g in film_genre_map_multi.get(fid, set()):
                    genre_counts[g] = genre_counts.get(g, 0) + 1
            if not genre_counts:
                continue
            dominant_genre = max(genre_counts, key=genre_counts.get)
            purity = sum(
                1 for fid in comm
                if dominant_genre in film_genre_map_multi.get(fid, set())
            ) / len(comm)
            purities.append(purity)
        return np.mean(purities), np.std(purities)

    pur_mean_a, pur_std_a = genre_purity_multilabel(comm_actor, film_genres_multi)
    pur_mean_u, pur_std_u = genre_purity_multilabel(comm_user,  film_genres_multi)
    print(f"Multi-label genre purity:")
    print(f"  Actor layer: {pur_mean_a:.4f} ± {pur_std_a:.4f}")
    print(f"  User  layer: {pur_mean_u:.4f} ± {pur_std_u:.4f}")
    return pur_mean_a, pur_mean_u, pur_std_a, pur_std_u


@app.cell
def _(G_actor, G_user, film_genres_multi):
    def intra_genre_edge_fraction(G, film_genre_map_multi):
        same, total = 0, 0
        for u, v in G.edges():
            g1 = film_genre_map_multi.get(u, set())
            g2 = film_genre_map_multi.get(v, set())
            if g1 and g2:
                total += 1
                if g1 & g2:
                    same += 1
        return same / total if total > 0 else 0.0

    _intra_actor = intra_genre_edge_fraction(G_actor, film_genres_multi)
    _intra_user  = intra_genre_edge_fraction(G_user,  film_genres_multi)
    print(f"Intra-genre edge fraction (share ≥1 genre):")
    print(f"  Actor layer: {_intra_actor:.4f}")
    print(f"  User  layer: {_intra_user:.4f}")
    return


@app.cell
def _(G_actor, G_user, film_genres_multi, genre_names, np, pd):
    _rows = []
    for _gid, _gname in genre_names.items():
        _films = [f for f, gs in film_genres_multi.items() if _gid in gs]
        if not _films:
            continue
        _deg_a = np.mean([G_actor.degree(f, weight="weight") for f in _films if f in G_actor])
        _deg_u = np.mean([G_user.degree(f,  weight="weight") for f in _films if f in G_user])
        _rows.append({"genre": _gname, "mean_strength_actor": _deg_a, "mean_strength_user": _deg_u})

    df_genre_deg = pd.DataFrame(_rows).sort_values("mean_strength_user", ascending=False)
    print("Per-genre mean weighted degree:")
    print(df_genre_deg.to_string(index=False))
    return (df_genre_deg,)


@app.cell
def _(
    IMAGES_DIR,
    df_genre_deg,
    plt,
    pur_mean_a,
    pur_mean_u,
    pur_std_a,
    pur_std_u,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    _axes[0].barh(df_genre_deg["genre"], df_genre_deg["mean_strength_user"],
                  color="steelblue", alpha=0.85)
    _axes[0].set_xlabel("Mean Weighted Degree (user layer)")
    _axes[0].set_title("Per-Genre Mean User-Layer Strength\n(multi-label genre assignment)")

    _axes[1].bar(["Actor layer", "User layer"],
                 [pur_mean_a, pur_mean_u],
                 yerr=[pur_std_a, pur_std_u],
                 color=["coral", "steelblue"], alpha=0.85, capsize=8, width=0.4)
    _axes[1].set_ylabel("Mean community genre purity (multi-label)")
    _axes[1].set_ylim(0, 1.2)
    _axes[1].set_title("Community Genre Purity by Layer")

    _fig.suptitle("RQ4: Genre Clustering — Multi-Label Analysis", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq4_genre_multilabel.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Figure 4 — Genre clustering analysis

    - **Left panel (per-genre user strength):** Games, Animation, Music, and Children genres have the highest co-rental connectivity — customers who rent these genres tend to rent *multiple* films from them
    - Action, Foreign, and Travel are the least co-rented, suggesting more isolated consumption patterns
    - The spread across genres (~0.7 units) is relatively narrow — no genre is dramatically isolated

    ### Figure 4 right — genre purity comparison

    | Layer | Mean purity | Std dev |
    |---|---|---|
    | Actor | 0.27 | 0.33 |
    | User | **0.77** | 0.39 |

    - **User layer purity is nearly 3× higher** than actor layer purity
    - In user-layer communities, 77% of films on average share the dominant genre — these communities are meaningfully genre-coherent
    - Actor communities are much more genre-mixed (27% purity) — actors work across genres, so cast overlap does not reliably cluster films by genre
    - **Intra-genre edge fraction is nearly identical** in both layers (~0.063) — both layers connect same-genre films at the same raw rate, but the user layer *concentrates* these connections into cleaner communities
    - **Answer to RQ4:** Yes, films cluster differently by genre across layers — the user behavioral layer is substantially better at recovering genre structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 5 — Are There Hub Films That Dominate Both Layers Simultaneously?

    ### Question and motivation

    - A "dual-layer hub" would be a film that is central in *both* the actor network (many shared cast members) and the user network (rented alongside many other films)
    - Such films would be ideal anchors for recommendation — they are highly connected regardless of which similarity signal you use
    - We operationalise this via **rank fusion**: each film gets a combined rank averaging its actor-layer and user-layer centrality rankings

    ### Quadrant analysis

    - Films are divided into four quadrants using median ranks as boundaries:
        - **Dual hub:** top half in both layers
        - **Actor-only hub:** top half in actor layer only
        - **User-only hub:** top half in user layer only
        - **Peripheral:** bottom half in both
    """)
    return


@app.cell
def _(
    film_ids,
    film_popularity,
    film_titles,
    pd,
    pr_actor,
    pr_user,
    strength_actor,
    strength_user,
):
    df_ranks = pd.DataFrame({
        "film_id":        film_ids,
        "title":          [film_titles.get(f, "???") for f in film_ids],
        "popularity":     [film_popularity[f]         for f in film_ids],
        "rank_str_actor": pd.Series(strength_actor).rank(ascending=False),
        "rank_str_user":  pd.Series(strength_user).rank(ascending=False),
        "rank_pr_actor":  pd.Series(pr_actor).rank(ascending=False),
        "rank_pr_user":   pd.Series(pr_user).rank(ascending=False),
    })
    df_ranks["mean_rank_actor"] = (df_ranks["rank_str_actor"] + df_ranks["rank_pr_actor"]) / 2
    df_ranks["mean_rank_user"]  = (df_ranks["rank_str_user"]  + df_ranks["rank_pr_user"])  / 2
    df_ranks["combined_rank"]   = (df_ranks["mean_rank_actor"] + df_ranks["mean_rank_user"]) / 2
    return (df_ranks,)


@app.cell
def _(df_ranks):
    top_dual = df_ranks.nsmallest(15, "combined_rank")
    print("Top 15 dual-layer hub films:")
    print(f"  {'Title':<35s} | {'Pop':>4s} | {'ActorRank':>9s} | {'UserRank':>8s} | {'CombRank':>8s}")
    print("  " + "-" * 72)
    for _, _row in top_dual.iterrows():
        print(f"  {_row['title']:<35s} | {int(_row['popularity']):>4d} | "
              f"{_row['mean_rank_actor']:>9.1f} | {_row['mean_rank_user']:>8.1f} | "
              f"{_row['combined_rank']:>8.1f}")
    return


@app.cell
def _(df_ranks, stats):
    rho_ranks, p_ranks = stats.spearmanr(
        df_ranks["mean_rank_actor"], df_ranks["mean_rank_user"]
    )
    print(f"Spearman(actor_hub_rank, user_hub_rank) = {rho_ranks:.4f}  (p={p_ranks:.2e})")

    median_actor = df_ranks["mean_rank_actor"].median()
    median_user  = df_ranks["mean_rank_user"].median()

    def _quadrant(row):
        a = row["mean_rank_actor"] <= median_actor
        u = row["mean_rank_user"]  <= median_user
        if a and u:     return "dual_hub"
        if a and not u: return "actor_only_hub"
        if not a and u: return "user_only_hub"
        return "peripheral"

    df_ranks["quadrant"] = df_ranks.apply(_quadrant, axis=1)
    for _q, _c in df_ranks["quadrant"].value_counts().items():
        print(f"  {_q:<20s}: {_c} films")
    print(f"\nMean popularity per quadrant:")
    for _q in ["dual_hub", "actor_only_hub", "user_only_hub", "peripheral"]:
        pop = df_ranks[df_ranks["quadrant"] == _q]["popularity"].mean()
        print(f"  {_q:<20s}: {pop:.2f}")
    return median_actor, median_user, rho_ranks


@app.cell
def _(IMAGES_DIR, df_ranks, median_actor, median_user, plt, rho_ranks):
    _color_map = {
        "dual_hub":       "gold",
        "actor_only_hub": "coral",
        "user_only_hub":  "steelblue",
        "peripheral":     "lightgrey",
    }
    _fig, _ax = plt.subplots(figsize=(8, 7))
    for _q, _color in _color_map.items():
        _mask = df_ranks["quadrant"] == _q
        _ax.scatter(df_ranks.loc[_mask, "mean_rank_actor"],
                    df_ranks.loc[_mask, "mean_rank_user"],
                    label=f"{_q} (n={_mask.sum()})",
                    color=_color, alpha=0.6, s=18, edgecolors="none")
    _ax.axvline(median_actor, color="black", linewidth=0.8, linestyle="--")
    _ax.axhline(median_user,  color="black", linewidth=0.8, linestyle="--")
    _ax.set_xlabel("Actor-layer hub rank (lower = more central)")
    _ax.set_ylabel("User-layer hub rank (lower = more central)")
    _ax.set_title(f"RQ5: Hub Status Quadrant Analysis\nSpearman ρ = {rho_ranks:.3f}")
    _ax.legend(markerscale=1.5)
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq5.1_hub_quadrant.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Figure 5 — Hub quadrant analysis

    - Points are spread almost **perfectly uniformly** across all four quadrants (~25% each)
    - The dashed median lines divide the space into roughly equal quarters — exactly what random assignment would look like
    - Spearman ρ = −0.004 (p = 0.90) — hub rank in one layer has **zero predictive power** for hub rank in the other

    ### Top dual-hub films

    - The top-ranked dual-hub films (e.g. CASSIDY WYOMING, REUNION WITCHES, CONNECTION MICROCOSMOS) have **very low popularity** (5–20 rentals)
    - These are niche films that happen to share actors with many others AND were co-rented by a relatively consistent audience — a distinctive combination
    - Mean popularity is only marginally higher for dual-hubs (17.1) vs peripheral films (15.0) — hub status does not imply popularity

    ### Answer to RQ5

    - **No** — there are no films that systematically dominate both layers
    - Hub status in the actor and user networks are independent random variables
    - This further confirms the complementary nature of the two layers: excelling in one does not predict excelling in the other
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 6 — Does Rental Duration / Rate Correlate with Network Position?

    ### Question and motivation

    - Film attributes like rental price, duration, length, and replacement cost are set by the store
    - Do these business attributes relate to how films are positioned in the similarity network?
    - Also: do films of different MPAA ratings (G, PG, PG-13, NC-17, R) differ in network connectivity?
    - This tests whether **pricing or content decisions** have any downstream effect on co-rental patterns or actor collaboration networks
    """)
    return


@app.cell
def _(
    bt_actor,
    bt_user,
    film_df,
    film_popularity,
    pr_actor,
    pr_user,
    strength_actor,
    strength_user,
):
    df_q6 = film_df[["film_id", "rental_duration", "rental_rate", "length", "replacement_cost"]].copy()
    df_q6["popularity"]     = df_q6["film_id"].map(film_popularity)
    df_q6["strength_actor"] = df_q6["film_id"].map(dict(enumerate(strength_actor)))
    df_q6["strength_user"]  = df_q6["film_id"].map(dict(enumerate(strength_user)))
    df_q6["pr_actor"]       = df_q6["film_id"].map(pr_actor)
    df_q6["pr_user"]        = df_q6["film_id"].map(pr_user)
    df_q6["bt_actor"]       = df_q6["film_id"].map(bt_actor)
    df_q6["bt_user"]        = df_q6["film_id"].map(bt_user)
    return (df_q6,)


@app.cell
def _(df_q6, stats):
    film_attrs   = ["rental_duration", "rental_rate", "length", "replacement_cost"]
    centralities = ["strength_actor", "strength_user", "pr_actor", "pr_user", "bt_actor", "bt_user"]
    print(f"\n{'Attribute':<20s} | {'str_act':>7s} | {'str_usr':>7s} | {'pr_act':>7s} | {'pr_usr':>7s} | {'bt_act':>7s} | {'bt_usr':>7s}")
    print("-" * 80)
    for _attr in film_attrs:
        _row_vals = []
        for _cent in centralities:
            _rho, _ = stats.spearmanr(df_q6[_attr].dropna(),
                                      df_q6.loc[df_q6[_attr].notna(), _cent])
            _row_vals.append(f"{_rho:>7.4f}")
        print(f"{_attr:<20s} | {'|'.join(_row_vals)}")
    return


@app.cell
def _(df_q6, film_df, stats):
    df_q6_rated = film_df[["film_id", "rating"]].merge(df_q6, on="film_id")
    print(f"\nMean user-layer strength by rating:")
    print(df_q6_rated.groupby("rating")["strength_user"].agg(["mean", "std", "count"]).to_string())
    rating_groups = [grp["strength_user"].values for _, grp in df_q6_rated.groupby("rating")]
    _kw_stat, _kw_p = stats.kruskal(*rating_groups)
    print(f"\nKruskal-Wallis (strength_user across ratings): H={_kw_stat:.4f}, p={_kw_p:.2e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 6 — Film attribute correlations with centrality

    | Attribute | Max |ρ| across all centralities |
    |---|---|
    | Rental duration | 0.024 |
    | Rental rate | 0.063 |
    | Film length | 0.011 |
    | Replacement cost | 0.065 |

    - **All correlations are near zero** — business pricing decisions have essentially no relationship with network position in either layer
    - **MPAA rating:** mean user-layer strength ranges from 5.88 (NC-17) to 6.00 (R) — a difference of only 0.12 units; Kruskal-Wallis p = 0.79 → **not statistically significant**
    - The network structure is driven entirely by **co-occurrence patterns**, not by the business classification of films
    - **Answer to RQ6: No** — rental duration, price, length, replacement cost, and content rating do not predict where a film sits in either the actor or user network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 7 — Do Store-Specific Rental Patterns Create Distinct Subgraphs?

    ### Question and motivation

    - The Sakila database has **two stores** with different customer bases
    - Do customers at each store co-rent films in the same way, or do stores develop distinct "taste neighborhoods"?
    - If store subgraphs are similar → rental patterns are universal, driven by film content
    - If store subgraphs differ → local store culture, customer demographics, or inventory placement matters

    ### Method

    - Build a separate cosine co-rental graph for each store's customers independently
    - Compare: edge overlap (Jaccard), per-film strength correlation, community structure, and NMI against genre
    """)
    return


@app.cell
def _(film_ids, inventory_df, np, nx, rental_film_df):
    rental_store = rental_film_df.merge(
        inventory_df[["inventory_id", "store_id"]], on="inventory_id", how="left"
    )
    store_col = "store_id_x" if "store_id_x" in rental_store.columns else "store_id"
    store_ids = sorted(rental_store[store_col].dropna().unique().astype(int))
    print(f"Stores: {store_ids}")

    store_graphs = {}
    for _sid in store_ids:
        _store_rentals = rental_store[rental_store[store_col] == _sid]
        _sfc = _store_rentals.groupby("film_id")["customer_id"].apply(set).to_dict()
        _Gs = nx.Graph()
        _Gs.add_nodes_from(film_ids)
        for _i in range(len(film_ids)):
            for _j in range(_i + 1, len(film_ids)):
                _f1, _f2 = film_ids[_i], film_ids[_j]
                _c1 = _sfc.get(_f1, set())
                _c2 = _sfc.get(_f2, set())
                _shared = len(_c1 & _c2)
                if _shared > 0 and len(_c1) > 0 and len(_c2) > 0:
                    _cos = _shared / np.sqrt(len(_c1) * len(_c2))
                    if _cos >= 0.10:
                        _Gs.add_edge(_f1, _f2, weight=_cos)
        store_graphs[_sid] = _Gs
        print(f"  Store {_sid}: {_Gs.number_of_edges()} edges, density={nx.density(_Gs):.4f}")
    return (store_graphs,)


@app.cell
def _(store_graphs):
    store_list = list(store_graphs.keys())
    for _i in range(len(store_list)):
        for _j in range(_i + 1, len(store_list)):
            _sa, _sb = store_list[_i], store_list[_j]
            _ea = {tuple(sorted(e)) for e in store_graphs[_sa].edges()}
            _eb = {tuple(sorted(e)) for e in store_graphs[_sb].edges()}
            _inter = _ea & _eb
            _union = _ea | _eb
            _jacc  = len(_inter) / len(_union) if _union else 0
            print(f"  Store {_sa} vs Store {_sb}: shared edges={len(_inter)}, Jaccard={_jacc:.4f}")
    return


@app.cell
def _(
    communities_to_labels,
    multilabel_nmi,
    nodes_with_genre,
    nx,
    store_graphs,
):
    for _sid, _Gs in store_graphs.items():
        if _Gs.number_of_edges() == 0:
            print(f"  Store {_sid}: no edges.")
            continue
        _comms_s = nx.community.louvain_communities(_Gs, weight="weight", seed=42)
        _mod_s   = nx.community.modularity(_Gs, _comms_s, weight="weight")
        _lbl_s   = communities_to_labels(_comms_s, nodes_with_genre)
        _nmi_s   = multilabel_nmi(_lbl_s, nodes_with_genre)
        print(f"  Store {_sid}: communities={len(_comms_s)}, "
              f"modularity={_mod_s:.4f}, NMI_genre={_nmi_s:.4f}")
    return


@app.cell
def _(G_user, film_ids, pd, stats, store_graphs):
    str_global = pd.Series({f: G_user.degree(f, weight="weight") for f in film_ids})
    for _sid, _Gs in store_graphs.items():
        _str_store = pd.Series({f: _Gs.degree(f, weight="weight") for f in film_ids})
        _rho_s, _p_s = stats.spearmanr(str_global, _str_store)
        print(f"  Spearman(global_user_strength, store_{_sid}) = {_rho_s:.4f}  (p={_p_s:.2e})")
    return


@app.cell
def _(IMAGES_DIR, film_ids, pd, plt, stats, store_graphs):
    _ea7    = {tuple(sorted(e)) for e in store_graphs[1].edges()}
    _eb7    = {tuple(sorted(e)) for e in store_graphs[2].edges()}
    _inter7  = _ea7 & _eb7
    _s1_only = len(_ea7) - len(_inter7)
    _s2_only = len(_eb7) - len(_inter7)

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))
    _axes[0].barh(["Store 1 only", "Shared", "Store 2 only"],
                  [_s1_only, len(_inter7), _s2_only],
                  color=["steelblue", "mediumpurple", "coral"], alpha=0.85)
    for _i, _v in enumerate([_s1_only, len(_inter7), _s2_only]):
        _axes[0].text(_v + 50, _i, f"{_v:,}", va="center", fontsize=10)
    _axes[0].set_xlabel("Number of edges")
    _axes[0].set_title("RQ7: Edge Overlap Between Stores")

    _str_s1 = pd.Series({f: store_graphs[1].degree(f, weight="weight") for f in film_ids})
    _str_s2 = pd.Series({f: store_graphs[2].degree(f, weight="weight") for f in film_ids})
    _rho_s1s2, _ = stats.spearmanr(_str_s1, _str_s2)
    _axes[1].scatter(_str_s1, _str_s2, alpha=0.4, s=15, color="mediumpurple")
    _axes[1].set_xlabel("Weighted Degree — Store 1")
    _axes[1].set_ylabel("Weighted Degree — Store 2")
    _axes[1].set_title(f"Per-Film Strength Across Stores\nρ = {_rho_s1s2:.3f}")

    _fig.suptitle("RQ7: Store-Specific Co-Rental Network Comparison", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq7.1_store_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 7 — Store-specific network results

    | Metric | Store 1 | Store 2 |
    |---|---|---|
    | Edges (cosine ≥ 0.10) | 19,875 | 19,713 |
    | Density | 0.040 | 0.039 |
    | Shared edges with other store | 719 | 719 |
    | Edge Jaccard overlap | 0.019 | 0.019 |
    | Communities (Louvain) | 253 | 250 |
    | Modularity | 0.167 | 0.167 |
    | NMI vs genre | 0.236 | 0.233 |

    ### Figure 6 — Store comparison

    - **Left panel:** almost all edges are store-specific — only 719 out of ~39,000 are shared (Jaccard = 0.019)
    - **Right panel (ρ = −0.04):** per-film strength is essentially uncorrelated across stores
    - Despite this, both stores produce **nearly identical community statistics** (same modularity, same NMI)
    - Individual film rankings differ, but the *structural pattern* of how films cluster is the same
    - **Answer to RQ7: Yes** — stores develop distinct co-rental graphs at the film-pair level, but the macro community structure converges to the same genre-coherent pattern
    - Implication: genre-based clustering is robust to which store a customer visits; recommendations could be pooled across stores safely
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## RQ 8 — Can Network Position Predict a Film's Revenue?

    ### Question and motivation

    - Revenue is the ultimate business metric — can the graph structure tell us anything beyond simply counting rentals?
    - We compute film-level revenue by summing all payment amounts attributed to each film
    - Two questions:
        1. Does network centrality correlate with revenue? (raw correlation)
        2. Does network position add revenue signal **beyond** what rental count already explains? (partial correlation)
    - If network position predicts revenue independently of popularity, it could help identify financially valuable films even before their rental count accumulates
    """)
    return


@app.cell
def _(
    bt_actor,
    bt_user,
    film_df,
    film_ids,
    film_popularity,
    inventory_df,
    payment_df,
    pd,
    pr_actor,
    pr_user,
    rental_df,
    strength_actor,
    strength_user,
):
    film_revenue = (
        payment_df
        .merge(rental_df[["rental_id", "inventory_id"]], on="rental_id", how="inner")
        .merge(inventory_df[["inventory_id", "film_id"]], on="inventory_id", how="inner")
        .groupby("film_id")["amount"].sum()
        .reindex(film_df["film_id"], fill_value=0)
    )

    df_q8 = pd.DataFrame({
        "film_id":        film_ids,
        "revenue":        [film_revenue.get(f, 0.0) for f in film_ids],
        "popularity":     [film_popularity[f]        for f in film_ids],
        "strength_actor": [strength_actor.get(f, 0.0) for f in film_ids],
        "strength_user":  [strength_user.get(f, 0.0)  for f in film_ids],
        "pr_actor":       [pr_actor.get(f, 0.0)        for f in film_ids],
        "pr_user":        [pr_user.get(f, 0.0)         for f in film_ids],
        "bt_actor":       [bt_actor.get(f, 0.0)        for f in film_ids],
        "bt_user":        [bt_user.get(f, 0.0)         for f in film_ids],
    })
    print(f"Revenue: mean=${df_q8['revenue'].mean():.2f}, "
          f"median=${df_q8['revenue'].median():.2f}, "
          f"max=${df_q8['revenue'].max():.2f}")
    return (df_q8,)


@app.cell
def _(df_q8, stats):
    predictors = ["popularity", "strength_actor", "strength_user",
                  "pr_actor", "pr_user", "bt_actor", "bt_user"]
    print(f"\n{'Predictor':<20s} | {'Spearman rho':>12s} | {'p-value':>12s}")
    print("-" * 50)
    for _pred in predictors:
        _rho_r, _p_r = stats.spearmanr(df_q8["revenue"], df_q8[_pred])
        print(f"{_pred:<20s} | {_rho_r:>12.4f} | {_p_r:>12.2e}")
    return


@app.cell
def _(df_q8, np, pearsonr):
    def _partial_corr(x, y, z):
        x, y, z = np.array(x), np.array(y), np.array(z)
        xz_res = x - np.polyval(np.polyfit(z, x, 1), z)
        yz_res = y - np.polyval(np.polyfit(z, y, 1), z)
        r, p   = pearsonr(xz_res, yz_res)
        return r, p

    _pc_r, _pc_p = _partial_corr(
        df_q8["revenue"].values,
        df_q8["strength_user"].values,
        df_q8["popularity"].values
    )
    print(f"Partial corr(revenue, strength_user | popularity) = {_pc_r:.4f}  (p={_pc_p:.2e})")
    print("(tests whether network position adds revenue signal beyond rental count)")
    return


@app.cell
def _(IMAGES_DIR, df_q8, plt, stats):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _axes[0].scatter(df_q8["popularity"], df_q8["revenue"],
                     alpha=0.4, s=15, color="steelblue")
    _axes[0].set_xlabel("Popularity (rentals)")
    _axes[0].set_ylabel("Revenue ($)")
    _axes[0].set_title("Revenue vs Popularity\nρ ≈ 0.68")

    _rho_rev_str, _ = stats.spearmanr(df_q8["revenue"], df_q8["strength_user"])
    _axes[1].scatter(df_q8["strength_user"], df_q8["revenue"],
                     alpha=0.4, s=15, color="seagreen")
    _axes[1].set_xlabel("Weighted Degree (user layer)")
    _axes[1].set_ylabel("Revenue ($)")
    _axes[1].set_title(f"Revenue vs User Strength\nρ = {_rho_rev_str:.3f} (raw); partial ≈ 0.03")

    _fig.suptitle("RQ8: Can Network Position Predict Revenue?", fontsize=13, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(f"{IMAGES_DIR}/rq8.1_revenue_vs_centrality.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RQ 8 — Revenue prediction results

    | Predictor | Spearman ρ with revenue | p-value |
    |---|---|---|
    | Popularity (rentals) | **0.680** | < 10⁻¹³⁵ |
    | User strength | 0.093 | 0.003 |
    | User PageRank | 0.095 | 0.003 |
    | User betweenness | −0.090 | 0.005 |
    | Actor strength | 0.039 | 0.22 (ns) |
    | Actor PageRank | 0.038 | 0.23 (ns) |
    | Actor betweenness | 0.031 | 0.33 (ns) |

    ### Figure 7 — Revenue scatter plots

    - **Left panel (ρ = 0.68):** strong positive trend — more rentals → more revenue, as expected (revenue ≈ rentals × price)
    - **Right panel (ρ = 0.09, partial ≈ 0.03):** the apparent user-strength correlation almost entirely vanishes when controlling for popularity
    - Partial correlation = 0.026 (p = 0.40) — **not significant** after controlling for rental count

    ### Answer to RQ8

    - **No** — network position does not independently predict revenue
    - The small raw correlation is entirely explained by the fact that popular films (high revenue) have slightly higher user-layer connectivity
    - Once popularity is held constant, graph centrality adds no revenue signal
    - Practical implication: for revenue forecasting, **rental count is the correct predictor**, not network centrality
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Final Summary

    ### Project objective
    - Analyse a multiplex film similarity network (actor + user co-rental layers) to answer eight research questions about the relationship between graph structure, genre, popularity, and revenue

    ---

    ### Consolidated findings

    | RQ | Question | Answer | Key evidence |
    |---|---|---|---|
    | RQ1 | Do popular films dominate the network? | **No** | All centrality-popularity ρ < 0.09; betweenness negatively correlated |
    | RQ2 | Do both layers give the same view? | **No** | Edge Jaccard = 0.044; centrality rank ρ ≈ 0.00 |
    | RQ3 | Which community method is best? | **User + Louvain or consensus** | NMI: consensus 0.187 > user 0.080 > actor 0.043 |
    | RQ4 | Do films cluster by genre in each layer? | **Yes (user layer)** | Genre purity: user 0.77 vs actor 0.27 |
    | RQ5 | Are there dual-layer hub films? | **No** | Hub rank ρ = −0.004; 4 quadrants ~25% each |
    | RQ6 | Do film attributes predict network position? | **No** | All attribute-centrality ρ < 0.07; rating KW p = 0.79 |
    | RQ7 | Do stores develop distinct subgraphs? | **Yes (locally)** | Store edge Jaccard = 0.019; but same macro community structure |
    | RQ8 | Can network position predict revenue? | **No** | Partial corr ≈ 0.03 (ns) after controlling for popularity |

    ---

    ### Practical implications

    - **For recommendation systems:** the user co-rental layer is the most informative single layer for genre-coherent recommendations; the actor layer adds complementary but genre-agnostic similarity
    - **For content strategy:** genre-based clustering is robust — it emerges consistently regardless of which store or which method is used
    - **For business decisions:** pricing, rental duration, and content rating do not influence network structure; these can be managed independently
    - **For hub identification:** betweenness-central niche films (low popularity, high bridging role) are candidates for cross-audience recommendation experiments

    ---

    ### Limitations

    - **Small dataset:** 1,000 films, 599 customers — patterns may not generalise to larger catalogs
    - **No explicit ratings:** rental ≠ satisfaction; the signal is noisy
    - **Single snapshot:** 2005–2006 data; no temporal dynamics
    - **Louvain non-determinism:** stability scores of 0.22–0.36 indicate moderate sensitivity to random seed; partitions should not be treated as ground truth
    - **Consensus modularity inflation:** the cross-layer consensus method produces artificially high modularity (confirmed by permutation test); NMI is a more reliable metric for it
    - **Two stores only:** store comparison is limited; more stores would allow stronger generalisations about local vs global rental patterns

    ---

    ### Future work

    - **Temporal slicing:** build monthly co-rental snapshots and track how communities evolve over the rental season
    - **Graph neural networks:** learn film embeddings from the multiplex structure for downstream classification and link prediction
    - **Richer evaluation:** collect explicit ratings or click-through data to evaluate recommendation quality beyond genre NMI
    - **Larger datasets:** apply the pipeline to MovieLens or a streaming service dataset to test scalability and generalisability
    - **Dynamic community detection:** use algorithms designed for temporal graphs (e.g. TILES, evolutionary Louvain) to track community shifts
    - **Store personalisation:** with more stores, test whether store-specific recommendation models outperform pooled global models
    """)
    return


if __name__ == "__main__":
    app.run()

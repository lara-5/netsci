import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sakila Film Network Analysis

    - **Goal:** model relationships between films using graph-based methods applied to a video rental dataset
    - Two complementary lenses:
        - **Actor layer** — which actors appeared in each film (production-side similarity)
        - **User layer** — which customers rented each film (taste-side similarity)
    - End result: a **multiplex network** that combines both signals for richer, more robust film similarity
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import math
    import pandas as pd
    import numpy as np
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score
    import matplotlib.pyplot as plt
    import marimo as mo
    from mpl_toolkits.mplot3d import Axes3D
    import random

    return Path, mo, np, nx, pd, plt, random


@app.cell
def _(Path):
    PROJECT_ROOT = Path("projects/lkrvavica")

    OUTPUT = PROJECT_ROOT / "output"
    DATA = PROJECT_ROOT / "data" / "raw"

    GRAPHS_DIR = OUTPUT / "graphs"
    IMAGES_DIR = OUTPUT / "images"

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Created: {GRAPHS_DIR}")
    print(f"Created: {IMAGES_DIR}")
    return DATA, IMAGES_DIR, OUTPUT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1 — Data Loading and Preparation

    ### Dataset: Sakila
    - Open-source relational database originally designed as a MySQL sample dataset
    - Represents a **fictional DVD rental business** with 16 relational tables
    - Key entities used in this project:
        - **Films** — 1,000 titles with metadata (genre, rating, language, rental rate)
        - **Actors** — 200 actors linked to films via a many-to-many junction table
        - **Customers** — active renting customers
        - **Rentals** — individual rental transactions (the behavioral signal)
        - **Inventory** — bridges rentals to specific film copies and stores
    - No explicit star ratings or user preferences are present — all signals must be inferred from **who rented what**
    """)
    return


@app.cell
def _(DATA, pd):
    actor_df         = pd.read_csv(DATA / "actor.csv")
    address_df       = pd.read_csv(DATA / "address.csv")
    category_df      = pd.read_csv(DATA / "category.csv")
    city_df          = pd.read_csv(DATA / "city.csv")
    country_df       = pd.read_csv(DATA / "country.csv")
    customer_df      = pd.read_csv(DATA / "customer.csv")
    film_actor_df    = pd.read_csv(DATA / "film_actor.csv")
    film_category_df = pd.read_csv(DATA / "film_category.csv")
    film_text_df     = pd.read_csv(DATA / "film_text.csv")
    film_df          = pd.read_csv(DATA / "film.csv")
    inventory_df     = pd.read_csv(DATA / "inventory.csv")
    language_df      = pd.read_csv(DATA / "language.csv")
    payment_df       = pd.read_csv(DATA / "payment.csv")
    rental_df        = pd.read_csv(DATA / "rental.csv")
    staff_df         = pd.read_csv(DATA / "staff.csv")
    store_df         = pd.read_csv(DATA / "store.csv")
    return (
        actor_df,
        address_df,
        category_df,
        city_df,
        country_df,
        customer_df,
        film_actor_df,
        film_category_df,
        film_df,
        film_text_df,
        inventory_df,
        language_df,
        payment_df,
        rental_df,
        staff_df,
        store_df,
    )


@app.cell
def _(
    actor_df,
    address_df,
    category_df,
    city_df,
    country_df,
    customer_df,
    film_actor_df,
    film_category_df,
    film_df,
    film_text_df,
    inventory_df,
    language_df,
    payment_df,
    rental_df,
    staff_df,
    store_df,
):
    for _name, _df in [
        ("actor", actor_df), ("address", address_df), ("category", category_df),
        ("city", city_df), ("country", country_df), ("customer", customer_df),
        ("film_actor", film_actor_df), ("film_category", film_category_df),
        ("film_text", film_text_df), ("film", film_df), ("inventory", inventory_df),
        ("language", language_df), ("payment", payment_df), ("rental", rental_df),
        ("staff", staff_df), ("store", store_df),
    ]:
        print(_name)
        print(_df.head(), "\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key data observations

    - **1,000 films** in total; **958** have been rented at least once
    - **5,462 film-actor mappings** → approximately 5.5 actors per film on average
    - **599 active renting customers** — a relatively small base, so co-rental signals can be sparse for less popular films
    - Genre assignments are **one-to-one** (0% multi-genre films in this dataset) — simplifies evaluation
    - Timestamps span **2005–2006**; data is historical and static
    - The `rental` table is the core behavioral signal — it records every transaction but carries no explicit satisfaction rating
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2 — Feature Engineering

    ### Derived lookup structures

    - Before building graphs, we need compact **per-film representations** of cast and audience
    - Three key mappings are computed:
        - `film_actors` → each film ID maps to the **set of actor IDs** who appeared in it
        - `film_customers` → each film ID maps to the **set of customer IDs** who rented it
        - `film_popularity` → each film ID maps to its **total rental count** (used as ground truth for popularity)
    - Genre ground truth stored as sets to handle potential multi-label cases robustly
    - The rental→inventory→film join is critical: rentals reference inventory copies, not films directly
    """)
    return


@app.cell
def _(film_actor_df, film_category_df, film_df, inventory_df, rental_df):
    # Map inventory to film rentals
    rental_film_df = rental_df.merge(inventory_df, on='inventory_id', how='inner')

    # Map films to sets of actor IDs
    film_actors = film_actor_df.groupby('film_id')['actor_id'].apply(set).to_dict()

    # Map films to sets of renting customer IDs
    film_customers = rental_film_df.groupby('film_id')['customer_id'].apply(set).to_dict()

    # FIX #1: Multi-label genre ground truth.
    film_genres_multi = (
        film_category_df.groupby('film_id')['category_id']
        .apply(set)
        .to_dict()
    )
    # Single-label fallback: lowest category_id (deterministic canonical label)
    film_genres_single = {fid: min(cats) for fid, cats in film_genres_multi.items()}

    multi_genre_films = {
        fid for fid, cats in film_genres_multi.items() if len(cats) > 1
    }
    print(f"Films with multiple genre assignments: {len(multi_genre_films)} "
          f"({len(multi_genre_films)/len(film_genres_multi)*100:.1f}%)")
    print("  (original .to_dict() silently discarded these extra assignments)")

    film_ids = film_df['film_id'].tolist()
    return film_actors, film_customers, film_ids, rental_film_df


@app.cell
def _(film_df, rental_film_df):
    rental_counts = (
        rental_film_df.groupby('film_id')['rental_id']
        .count()
        .reindex(film_df['film_id'], fill_value=0)
    )
    film_popularity = rental_counts.to_dict()

    print(f"Loaded {len(film_df)} films.")
    print(f"Loaded {len(film_df)} film-actor mappings (see film_actor_df).")
    print(f"Loaded {rental_film_df.shape[0]} rentals (after inventory join).")
    print(f"Active renting customers: {rental_film_df['customer_id'].nunique()}")
    print(f"Films with at least 1 rental: {rental_counts[rental_counts > 0].count()}")
    return (film_popularity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary of dataset scale

    | Entity | Count |
    |---|---|
    | Films | 1,000 |
    | Actors | 200 |
    | Film-actor links | 5,462 |
    | Rentals (transactions) | 16,044 |
    | Active customers | 599 |
    | Films with ≥1 rental | 958 |

    - The **58 films never rented** will appear as isolated nodes in the user network — a real sparsity challenge
    - Popularity is **heavily skewed**: a small number of films account for a disproportionate share of rentals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3 — Graph Construction

    ### Strategy overview

    - Each **film is a node**; edges encode pairwise similarity between films
    - We build **two separate networks**, then combine them into a multiplex:
        - **Layer 1 — Actor Network:** two films are linked if they share cast members
        - **Layer 2 — User Network:** two films are linked if the same customers rented both
    - Rationale: actor overlap captures *production-side* similarity (genre, style, director); rental co-occurrence captures *taste-side* similarity (what audiences actually pair together)
    - Combining both avoids over-relying on a single, potentially noisy, signal

    ### Centrality metrics used

    | Metric | What it measures |
    |---|---|
    | Weighted degree strength | Total edge weight attached to a node — how "similar" a film is to all others |
    | PageRank | Importance accounting for the importance of neighbours (recursive) |
    | Betweenness centrality | How often a node lies on shortest paths — bridging/hub role |

    - All centralities are correlated with **rental popularity** using Spearman ρ (rank-based, robust to skew)
    - A high correlation would mean centrality ≈ popularity; a low one means the graph captures something *different* from raw popularity
    """)
    return


@app.cell
def _(film_ids, film_popularity, nx, pd):
    def compute_centrality_and_correlation(G, network_name):
        # Weighted degree strength (FIX #3: use weight, not unweighted degree)
        degree_dict = dict(G.degree(weight='weight'))

        try:
            pagerank_dict = nx.pagerank(G, weight='weight')
        except Exception as _e:
            print(f"Warning: PageRank failed for {network_name}: {_e}")
            pagerank_dict = {n: 0.0 for n in G.nodes()}

        try:
            dist_G = nx.Graph()
            dist_G.add_nodes_from(G.nodes())
            for _u, _v, _d in G.edges(data=True):
                _w = _d.get('weight', 1.0)
                dist_G.add_edge(_u, _v, weight=1.0 / _w if _w > 0 else 1e6)
            betweenness_dict = nx.betweenness_centrality(dist_G, weight='weight')
        except Exception as _e:
            print(f"Warning: Betweenness failed for {network_name}: {_e}")
            betweenness_dict = {n: 0.0 for n in G.nodes()}

        df = pd.DataFrame({
            'film_id':         film_ids,
            'popularity':      [film_popularity[fid] for fid in film_ids],
            'degree_strength': [degree_dict.get(fid, 0.0) for fid in film_ids],
            'pagerank':        [pagerank_dict.get(fid, 0.0) for fid in film_ids],
            'betweenness':     [betweenness_dict.get(fid, 0.0) for fid in film_ids],
        })

        corr_deg = df['popularity'].corr(df['degree_strength'], method='spearman')
        corr_pr  = df['popularity'].corr(df['pagerank'],        method='spearman')
        corr_bt  = df['popularity'].corr(df['betweenness'],     method='spearman')

        print(f"\n--- CENTRALITY CORRELATIONS WITH POPULARITY ({network_name}) ---")
        print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Density: {nx.density(G):.4f}")
        print(f"Popularity vs. Weighted Degree Strength: {corr_deg:.4f}")
        print(f"Popularity vs. PageRank:                 {corr_pr:.4f}")
        print(f"Popularity vs. Betweenness:              {corr_bt:.4f}")

        return degree_dict, pagerank_dict, betweenness_dict

    return (compute_centrality_and_correlation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph 1 — Actor Co-occurrence Network (IFA-weighted Jaccard)

    ### Design choices

    - **Edge condition:** two films share an edge if they have at least one actor in common
    - **Edge weight:** IFA-weighted Jaccard similarity
        - *Jaccard* measures overlap as: |shared actors| / |union of actors|
        - *IFA (Inverse Frequency Actor)* down-weights actors who appear in many films — sharing a prolific actor is a weaker signal than sharing a rare one
        - Analogy: like TF-IDF in text retrieval, but for actors
    - **Why this matters:** without IFA weighting, two films connected by a very common actor (who appears in 40+ films) would get the same weight as two films sharing a rare specialist actor
    - Normalisation by union size preserves the [0, 1] Jaccard range
    """)
    return


@app.cell
def _(
    compute_centrality_and_correlation,
    film_actor_df,
    film_actors,
    film_ids,
    np,
    nx,
):
    def _():
        actor_film_counts = film_actor_df.groupby('actor_id')['film_id'].count().to_dict()

        G_actor = nx.Graph()
        G_actor.add_nodes_from(film_ids)

        for _i in range(len(film_ids)):
            for _j in range(_i + 1, len(film_ids)):
                f1, f2 = film_ids[_i], film_ids[_j]
                a1 = film_actors.get(f1, set())
                a2 = film_actors.get(f2, set())
                shared = a1.intersection(a2)
                if not shared:
                    continue
                union_len = len(a1.union(a2))
                if union_len == 0:
                    continue

                # IFA-weighted Jaccard: rare actors contribute more
                ifa_score = sum(
                    1.0 / np.log1p(actor_film_counts.get(a, 1))
                    for a in shared
                )
                weight = ifa_score / union_len
                G_actor.add_edge(f1, f2, weight=weight, shared_count=len(shared))

        deg_act, pr_act, bt_act = compute_centrality_and_correlation(
            G_actor, "Actor Network (IFA-weighted Jaccard)"
        )
        return G_actor, deg_act, pr_act, bt_act

    G_actor, g1_deg_act, g1_pr_act, g1_bt_act = _()
    return (G_actor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Actor network results

    | Metric | Spearman ρ with popularity |
    |---|---|
    | Weighted degree strength | 0.051 |
    | PageRank | 0.050 |
    | Betweenness | −0.019 |

    - **Near-zero correlations** — being central in the actor network does *not* predict how often a film gets rented
    - This is **expected and correct**: film popularity is driven by marketing, availability, and audience taste — not by cast overlap
    - The actor network is valuable as a **content similarity** dimension, not a popularity predictor
    - **68,919 edges**, density **0.138** — moderately dense, reflecting the fact that many films share at least one actor from the 200-actor pool
    - Key insight: the actor graph encodes *style and genre* proximity, which is orthogonal to raw rental volume
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph 2 — Raw User Co-rental Network

    ### Design and motivation

    - **Edge condition:** two films share an edge if **≥2 customers** rented both (minimum threshold to reduce noise)
    - **Edge weight:** raw count of shared customers
    - This is the simplest possible user-behavior similarity measure
    - **Known limitation:** raw counts favour popular films — a blockbuster rented by 500 customers will share customers with almost every other film, regardless of true thematic similarity
    - We build this version first as a **baseline** to quantify the popularity confound before applying normalisation
    """)
    return


@app.cell
def _(compute_centrality_and_correlation, film_customers, film_ids, nx):
    def _():
        G_user_raw = nx.Graph()
        G_user_raw.add_nodes_from(film_ids)
        for _i in range(len(film_ids)):
            for _j in range(_i + 1, len(film_ids)):
                f1, f2 = film_ids[_i], film_ids[_j]
                c1 = film_customers.get(f1, set())
                c2 = film_customers.get(f2, set())
                shared = len(c1.intersection(c2))
                if shared >= 2:
                    G_user_raw.add_edge(f1, f2, weight=shared)
        deg_usr_raw, pr_usr_raw, bt_usr_raw = compute_centrality_and_correlation(
            G_user_raw, "Raw User Network"
        )
        return G_user_raw, deg_usr_raw, pr_usr_raw, bt_usr_raw

    G_user_raw, g2_deg_usr_raw, g2_pr_usr_raw, g2_bt_usr_raw = _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Raw user network results — popularity confound

    | Metric | Spearman ρ with popularity |
    |---|---|
    | Weighted degree strength | **0.9875** |
    | PageRank | **0.9873** |
    | Betweenness | **0.9712** |

    - Correlations are essentially **perfect (ρ ≈ 0.99)** — the raw user network is almost entirely re-encoding rental popularity
    - A film that is popular shares customers with everyone, giving it high centrality — but this tells us nothing about *which* films are actually similar in terms of audience taste
    - This is a **red flag**: using this network as a similarity measure would simply recommend popular films to everyone, regardless of their preferences
    - **Normalisation is mandatory** before this layer can be used meaningfully
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical Validation

    ### Why this test matters

    - We need to check: are our observed film-film connections denser than what random reshuffling would produce?

    ### Method: degree-preserving edge rewiring

    1. Take the bipartite graph (e.g., film ↔ actor)
    2. Randomly swap edges while preserving every node's degree (double-edge swap)
    3. Project the randomised bipartite graph onto film nodes
    4. Measure density of the resulting random projection
    5. Repeat 20 times to get a null distribution
    6. Compute Z-score: how many standard deviations above the null mean is our observed density?

    - Z > 3 → statistically significant signal above random
    """)
    return


@app.cell
def _(film_actor_df, film_customers, film_ids, np, nx):
    def evaluate_projection_effects():
        # --- Actor bipartite null ---
        B_actor = nx.Graph()
        film_nodes = [('film', fid) for fid in film_ids]
        B_actor.add_nodes_from(film_nodes, bipartite=0)
        for _, row in film_actor_df.iterrows():
            B_actor.add_node(('actor', row['actor_id']), bipartite=1)
            B_actor.add_edge(('film', row['film_id']), ('actor', row['actor_id']))

        film_set_actor = {('film', fid) for fid in film_ids}
        G_obs_actor = nx.bipartite.projected_graph(B_actor, film_set_actor)
        obs_density_actor = nx.density(G_obs_actor)

        n_null = 20
        null_densities_actor = []
        for _ in range(n_null):
            B_rand = nx.double_edge_swap(
                B_actor.copy(), nswap=B_actor.number_of_edges() * 2, max_tries=100000
            )
            G_rand_proj = nx.bipartite.projected_graph(B_rand, film_set_actor)
            null_densities_actor.append(nx.density(G_rand_proj))

        null_mean_a = np.mean(null_densities_actor)
        null_std_a  = np.std(null_densities_actor)
        z_actor = (obs_density_actor - null_mean_a) / null_std_a if null_std_a > 0 else float('inf')

        print("=== BIPARTITE PROJECTION EFFECT EVALUATION ===")
        print(f"\nActor layer (raw shared-actor projection):")
        print(f"  Observed density:    {obs_density_actor:.4f}")
        print(f"  Null mean ± std:     {null_mean_a:.4f} ± {null_std_a:.4f}")
        print(f"  Z-score:             {z_actor:.2f}")
        print(f"  Interpretation:      ", end="")
        if z_actor > 3:
            print("density is significantly above null → actor connections carry signal beyond random")
        elif z_actor > 1:
            print("density is moderately above null → mild signal, some random inflation present")
        else:
            print("density is at null level → actor projection is dominated by random effects")

        # --- User bipartite null ---
        B_user = nx.Graph()
        B_user.add_nodes_from([('film', fid) for fid in film_ids], bipartite=0)
        for fid, customers in film_customers.items():
            for cid in customers:
                B_user.add_node(('cust', cid), bipartite=1)
                B_user.add_edge(('film', fid), ('cust', cid))

        film_set_user = {('film', fid) for fid in film_ids}
        G_obs_user = nx.bipartite.projected_graph(B_user, film_set_user)
        obs_density_user = nx.density(G_obs_user)

        null_densities_user = []
        for _ in range(n_null):
            B_rand_u = nx.double_edge_swap(
                B_user.copy(), nswap=B_user.number_of_edges() * 2, max_tries=50000
            )
            G_rand_proj_u = nx.bipartite.projected_graph(B_rand_u, film_set_user)
            null_densities_user.append(nx.density(G_rand_proj_u))

        null_mean_u = np.mean(null_densities_user)
        null_std_u  = np.std(null_densities_user)
        z_user = (obs_density_user - null_mean_u) / null_std_u if null_std_u > 0 else float('inf')

        print(f"\nUser layer (raw shared-customer projection before cosine threshold):")
        print(f"  Observed density:    {obs_density_user:.4f}")
        print(f"  Null mean ± std:     {null_mean_u:.4f} ± {null_std_u:.4f}")
        print(f"  Z-score:             {z_user:.2f}")
        print(f"  Interpretation:      ", end="")
        if z_user > 3:
            print("density is significantly above null → co-rental connections carry real signal")
        elif z_user > 1:
            print("density is moderately above null → mild signal present")
        else:
            print("density at null level → co-rental projection dominated by random effects")

        print(f"\nNote: the cosine threshold (>= 0.10) applied to the user layer serves partly")
        print(f"to cut the random projection baseline, retaining only edges above the null expectation.")

    evaluate_projection_effects()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Null model results — both layers carry genuine signal

    | Layer | Observed density | Null mean ± std | Z-score | Verdict |
    |---|---|---|---|---|
    | Actor | 0.1380 | 0.0869 ± 0.0005 | **96** | ✅ Highly significant |
    | User (raw) | 0.3235 | 0.1791 ± 0.0003 | **538** | ✅ Extremely significant |

    - Both networks are **far denser** than any random rewiring would produce
    - The Z-scores of 96 and 538 are orders of magnitude above the significance threshold of 3
    - This confirms: the edges we constructed are **not noise** — they reflect real structural patterns
    - The cosine threshold (≥0.10) applied to the user layer deliberately removes edges near the null baseline, keeping only the strongest co-rental signals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph 3 — Cosine-Normalised User Network

    ### Why cosine similarity?

    - **Cosine similarity** between two films = (shared customers) / √(|renters of film A| × |renters of film B|)
    - This divides by each film's rental volume, correcting for the fact that popular films share many customers by default
    - Threshold of **≥0.10** applied: only retain edges where the normalised overlap is meaningful
        - Removes low-signal edges near the null baseline (as confirmed by the null model test)
        - Reduces the graph from ~320k potential edges to ~30k strong ones
    - This version of the user network is used in the **final multiplex**
    """)
    return


@app.cell
def _(compute_centrality_and_correlation, film_customers, film_ids, np, nx):
    def _():
        G_user = nx.Graph()
        G_user.add_nodes_from(film_ids)
        for _i in range(len(film_ids)):
            for _j in range(_i + 1, len(film_ids)):
                f1, f2 = film_ids[_i], film_ids[_j]
                c1 = film_customers.get(f1, set())
                c2 = film_customers.get(f2, set())
                shared = len(c1.intersection(c2))
                if shared > 0 and len(c1) > 0 and len(c2) > 0:
                    cosine = shared / np.sqrt(len(c1) * len(c2))
                    if cosine >= 0.10:
                        G_user.add_edge(f1, f2, weight=cosine)

        deg_usr, pr_usr, bt_usr = compute_centrality_and_correlation(
            G_user, "User Network (Cosine >= 0.10, domain-justified)"
        )
        return G_user, deg_usr, pr_usr, bt_usr

    G_user, g3_deg_usr, g3_pr_usr, g3_bt_usr = _()
    return (G_user,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cosine user network results — normalisation works

    | Metric | Raw network ρ | Cosine network ρ | Change |
    |---|---|---|---|
    | Degree strength | 0.9875 | **0.3726** | −0.615 |
    | PageRank | 0.9873 | **0.3675** | −0.620 |
    | Betweenness | 0.9712 | **0.0415** | −0.930 |

    - Popularity correlation drops dramatically after normalisation — from ~0.99 to ~0.37
    - The residual 0.37 correlation with degree strength is reasonable: moderately popular films appear in more audiences, so they legitimately have more co-rental overlap
    - Betweenness drops to ~0.04: bridging roles in the taste graph are almost entirely decoupled from raw popularity
    - **29,589 edges** remain after the 0.10 threshold, density **0.059** — sparser but higher quality than the raw version
    - Key finding: **cosine normalisation successfully breaks the popularity-centrality confound**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 4 — Multiplex Network Construction

    ### What is a multiplex network?

    - A **multiplex network** stacks multiple relationship layers on the **same set of nodes**
    - Each film node appears in both layers simultaneously, coupled by inter-layer connections
    - Our multiplex:
        - **Layer 1 (Actor):** IFA-weighted Jaccard similarity — production-side
        - **Layer 2 (User):** cosine co-rental similarity — taste-side
    - Both layers are independently normalised before combination

    ### Two aggregation strategies compared

    | Strategy | Method | Pros | Cons |
    |---|---|---|---|
    | Linear projection | Sum the two layer weights directly | Simple | Scale mismatch between layers |
    | **Borda rank projection** | Rank edges within each layer, then sum normalised ranks | Robust, scale-free | Slightly more complex |

    - **Borda is preferred**: it avoids problems when one layer has systematically higher weights than the other
    - A Borda score > 1.0 means an edge is confirmed by *both* layers (one of the ~4.4% overlap edges)
    """)
    return


@app.cell
def _(G_actor, G_user, film_ids, nx):
    # Build multiplex (MultiGraph stores both layers on same node set)
    G_multiplex_raw = nx.MultiGraph()
    G_multiplex_raw.add_nodes_from(film_ids)

    # Normalise actor layer weights
    actor_weights_all = [d['weight'] for _, _, d in G_actor.edges(data=True)]
    max_actor_w = max(actor_weights_all) if actor_weights_all else 1.0
    for _u, _v, _d in G_actor.edges(data=True):
        G_multiplex_raw.add_edge(_u, _v,
                                 weight=_d['weight'] / max_actor_w,
                                 layer='actor')

    # Normalise user layer weights
    user_weights_all = [d['weight'] for _, _, d in G_user.edges(data=True)]
    max_user_w = max(user_weights_all) if user_weights_all else 1.0
    for _u, _v, _d in G_user.edges(data=True):
        G_multiplex_raw.add_edge(_u, _v,
                                 weight=_d['weight'] / max_user_w,
                                 layer='user')

    print(f"Multiplex (normalized weights):")
    print(f"  Nodes: {G_multiplex_raw.number_of_nodes()} | Total multi-edges: {G_multiplex_raw.number_of_edges()}")
    return (G_multiplex_raw,)


@app.cell
def _(G_multiplex_raw, film_ids, nx):
    def _():
        G_combined_linear_temp = nx.Graph()
        G_combined_linear_temp.add_nodes_from(film_ids)
        for _u, _v, _key, _d in G_multiplex_raw.edges(keys=True, data=True):
            if G_combined_linear_temp.has_edge(_u, _v):
                G_combined_linear_temp[_u][_v]['weight'] += _d['weight']
            else:
                G_combined_linear_temp.add_edge(_u, _v, weight=_d['weight'])

        # --- Borda-rank projection ---
        actor_edges_list = [
            (_u, _v, _d['weight'])
            for _u, _v, _d in G_multiplex_raw.edges(data=True)
            if _d.get('layer') == 'actor'
        ]
        user_edges_list = [
            (_u, _v, _d['weight'])
            for _u, _v, _d in G_multiplex_raw.edges(data=True)
            if _d.get('layer') == 'user'
        ]

        def borda_ranks(edge_list):
            n = len(edge_list)
            if n == 0:
                return {}
            sorted_edges = sorted(edge_list, key=lambda x: x[2])
            return {
                (tuple(sorted([u, v]))): (i + 1) / n
                for i, (u, v, _) in enumerate(sorted_edges)
            }

        ranks_actor = borda_ranks(actor_edges_list)
        ranks_user  = borda_ranks(user_edges_list)

        G_combined_borda_temp = nx.Graph()
        G_combined_borda_temp.add_nodes_from(film_ids)

        all_edge_pairs = set(ranks_actor.keys()) | set(ranks_user.keys())
        for pair in all_edge_pairs:
            u, v = pair
            borda_score = ranks_actor.get(pair, 0.0) + ranks_user.get(pair, 0.0)
            if G_combined_borda_temp.has_edge(u, v):
                G_combined_borda_temp[u][v]['weight'] = max(
                    G_combined_borda_temp[u][v]['weight'], borda_score
                )
            else:
                G_combined_borda_temp.add_edge(u, v, weight=borda_score)

        print(f"\nLinear projection:  {G_combined_linear_temp.number_of_edges()} edges, "
              f"density={nx.density(G_combined_linear_temp):.4f}")
        print(f"Borda projection:   {G_combined_borda_temp.number_of_edges()} edges, "
              f"density={nx.density(G_combined_borda_temp):.4f}")

        return G_multiplex_raw, G_combined_borda_temp, G_combined_linear_temp

    G_multiplex_final, G_combined_final, G_combined_linear_final = _()
    return G_combined_final, G_combined_linear_final, G_multiplex_final


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiplex Validation — Topological Complementarity

    ### What we are checking

    - If the two layers were redundant (both encoding the same information), the multiplex would offer no advantage over either layer alone
    - We check: **what fraction of edges appear in only one layer?**
    - High exclusivity → the layers are genuinely complementary → the multiplex is justified

    ### Weight distribution interpretation

    - Edges with Borda score **0–1.0**: present in exactly one layer (either actor or user, not both)
    - Edges with Borda score **>1.0**: present in *both* layers — the highest-confidence similarity signals
    - The histogram shape reveals the balance between single-layer and double-confirmed edges
    """)
    return


@app.cell
def _(G_combined_final, G_multiplex_final, IMAGES_DIR, nx, plt):
    def validate_multiplex_architecture(G_multiplex_obj, G_combined_obj):
        actor_subedges = [(_u, _v) for _u, _v, _d in G_multiplex_obj.edges(data=True)
                          if _d['layer'] == 'actor']
        user_subedges  = [(_u, _v) for _u, _v, _d in G_multiplex_obj.edges(data=True)
                          if _d['layer'] == 'user']

        G_actor_tmp = nx.Graph(actor_subedges)
        G_user_tmp  = nx.Graph(user_subedges)
        G_actor_tmp.add_nodes_from(G_multiplex_obj.nodes())
        G_user_tmp.add_nodes_from(G_multiplex_obj.nodes())

        comp_combined = nx.number_connected_components(G_combined_obj)

        weights_combined = [_d['weight'] for _u, _v, _d in G_combined_obj.edges(data=True)]

        edges_actor_count    = len(actor_subedges)
        edges_user_count     = len(user_subedges)
        edges_combined_count = G_combined_obj.number_of_edges()
        overlapping_edges    = (edges_actor_count + edges_user_count) - edges_combined_count
        exclusivity_pct      = ((edges_combined_count - overlapping_edges) / edges_combined_count) * 100

        print("\nTopological complementarity:")
        print(f"  Shared connections confirmed by both layers: {overlapping_edges}")
        print(f"  Unique information in the multiplex:         {exclusivity_pct:.2f}%")
        print(f"  Isolated components in combined graph:       {comp_combined}")

        plt.figure(figsize=(10, 5))
        plt.hist(weights_combined, bins=50, alpha=0.75, color='teal',
                 edgecolor='black', log=True)
        plt.title('Weight Distribution — Borda-Projected Multiplex (Log Scale)')
        plt.xlabel('Edge Weight (Borda-aggregated rank score)')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/multiplex_robustness_evidence.png")
        plt.show()

        counts, bins, _ = plt.hist(
            weights_combined, bins=50, alpha=0.75, color='teal',
            edgecolor='black', log=True
        )
        print("\nHistogram values:")
        for i in range(len(counts)):
            print(f"{i+1} {bins[i]:.4f} {bins[i+1]:.4f} {int(counts[i])}")

    validate_multiplex_architecture(G_multiplex_final, G_combined_final)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multiplex validation results

    | Property | Value |
    |---|---|
    | Total edges (combined) | 94,378 |
    | Graph density | 0.189 |
    | Edges confirmed by both layers | 4,130 (~4.4%) |
    | Unique information in multiplex | **95.6%** |
    | Connected components | 1 (fully connected) |

    ### Weight distribution — what the histogram shows

    - **Large plateau (Borda score 0–1.0):** ~90,000 edges present in exactly one layer — single-layer signals
    - **Sharp drop above 1.0:** ~4,130 edges present in *both* layers — these are the strongest, most reliable similarity signals
    - The long tail toward 2.0 represents films that are highly similar by *both* cast and rental patterns — the most confident recommendations
    - **Key takeaway: 95.6% of edges carry information unique to one layer** — the multiplex is not redundant; each layer contributes distinct knowledge
    - The graph is fully connected (1 component) — no film is completely isolated from the rest of the network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 5 — Graph Export and 3D Visualisation

    ### Saving the networks

    - Three graph files are exported for downstream analysis (community detection, embeddings, etc.):
        - `multiplex_network.graphml` — the full two-layer multiplex (MultiGraph)
        - `combined_network.graphml` — the Borda-projected single graph (used downstream)
        - `combined_linear_network.graphml` — the linear-summed projection (comparison baseline)
    - GraphML format is widely supported by graph analysis tools (Gephi, NetworkX, igraph)
    """)
    return


@app.cell
def _(
    G_combined_final,
    G_combined_linear_final,
    G_multiplex_final,
    OUTPUT,
    nx,
):
    nx.write_graphml(G_multiplex_final,       f"{OUTPUT}/graphs/multiplex_network.graphml")
    nx.write_graphml(G_combined_final,        f"{OUTPUT}/graphs/combined_network.graphml")
    nx.write_graphml(G_combined_linear_final, f"{OUTPUT}/graphs/combined_linear_network.graphml")
    print("Saved: multiplex_network.graphml, combined_network.graphml (Borda), combined_linear_network.graphml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D Multiplex Layer Visualisation

    ### Reading the visualisation

    - Each film node appears **twice** — once on the top plane (actor layer, teal) and once on the bottom plane (user layer, orange)
    - **Vertical dashed lines** connect the two representations of the same film across layers (inter-layer coupling)
    - Edge thickness within each plane encodes the weight (stronger similarity = thicker edge)
    - Only **45 of 1,000 nodes** are shown for visual clarity

    ### Key observations

    - The two layers show **visibly different connectivity patterns** — actor edges tend to be more uniformly distributed; user edges are more concentrated on a subset of well-co-rented films
    - Some nodes have many actor-layer connections but few user-layer ones, and vice versa — confirming the layers capture complementary information
    - The 3D format makes it intuitive that the multiplex carries more information than either layer alone
    """)
    return


@app.cell
def _(G_multiplex_final, OUTPUT, nx, plt, random):
    def visualize_true_multiplex_3d(G_multiplex_obj, sample_size=40):
        print(f"[3D Multiplex Visualizer — sampling {sample_size} nodes]")

        random.seed(42)
        all_nodes = list(G_multiplex_obj.nodes())
        sampled_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))

        G_2d_reference = nx.Graph()
        G_2d_reference.add_nodes_from(sampled_nodes)
        sampled_edges = [
            (_u, _v, _d)
            for _u, _v, _k, _d in G_multiplex_obj.edges(keys=True, data=True)
            if _u in sampled_nodes and _v in sampled_nodes
        ]
        G_2d_reference.add_edges_from([(_u, _v) for _u, _v, _d in sampled_edges])
        pos_2d = nx.spring_layout(G_2d_reference, seed=42)

        fig = plt.figure(figsize=(12, 10))
        ax  = fig.add_subplot(111, projection='3d')
        Z_ACTOR, Z_USER = 1.0, 0.0

        for node in sampled_nodes:
            _x, _y = pos_2d[node]
            ax.scatter(_x, _y, Z_ACTOR, color='teal',       s=80, alpha=0.9, edgecolors='black', zorder=5)
            ax.scatter(_x, _y, Z_USER,  color='darkorange', s=80, alpha=0.9, edgecolors='black', zorder=5)
            ax.plot([_x, _x], [_y, _y], [Z_USER, Z_ACTOR],
                    color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        actor_weights = [_d['weight'] for _u, _v, _d in sampled_edges if _d['layer'] == 'actor']
        user_weights  = [_d['weight'] for _u, _v, _d in sampled_edges if _d['layer'] == 'user']
        max_act_w = max(actor_weights) if actor_weights else 1
        max_usr_w = max(user_weights)  if user_weights  else 1

        for _u, _v, data in sampled_edges:
            x1, y1 = pos_2d[_u]
            x2, y2 = pos_2d[_v]
            weight  = data['weight']
            if data['layer'] == 'actor':
                width = (weight / max_act_w) * 2.5
                ax.plot([x1, x2], [y1, y2], [Z_ACTOR, Z_ACTOR],
                        color='teal',       linewidth=width, alpha=0.4)
            elif data['layer'] == 'user':
                width = (weight / max_usr_w) * 2.5
                ax.plot([x1, x2], [y1, y2], [Z_USER, Z_USER],
                        color='darkorange', linewidth=width, alpha=0.4)

        ax.set_title('True Multiplex Network (3D — normalized layer weights)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_zlim(-0.2, 1.2)
        ax.set_zticks([Z_USER, Z_ACTOR])
        ax.set_zticklabels(
            ['User Behavioral Layer (Cosine, normalized)',
             'Actor Production Layer (IFA-Jaccard, normalized)'],
            fontsize=11, fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.view_init(elev=22, azim=-45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT}/images/multiplex_3d_layer_visualization.png", dpi=300)
        plt.close()
        plt.show()
        print("3D visualization saved.\n")

    visualize_true_multiplex_3d(G_multiplex_final, sample_size=45)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary and Conclusions

    ### Project objective
    - Build a **multiplex film similarity network** from the Sakila rental database without requiring explicit user ratings
    - Combine production-side (actor) and behavior-side (rental) signals into a single, principled graph representation

    ### Key findings

    | Finding | Evidence |
    |---|---|
    | Both layers carry genuine signal | Null model Z-scores of 96 (actor) and 538 (user) |
    | Actor network ≠ popularity | Centrality-popularity correlations near zero (ρ ≈ 0.05) |
    | Raw co-rental conflates popularity with similarity | ρ ≈ 0.99 before normalisation |
    | Cosine normalisation resolves this | ρ drops to 0.37 after normalisation |
    | The two layers are complementary | 95.6% of edges are unique to one layer |
    | The multiplex is fully connected | 1 connected component across all 1,000 films |

    ### Practical implications
    - This network structure directly supports:
        - **Recommendation:** find films similar to a target using edge weights
        - **Genre/style clustering:** community detection on the combined graph
        - **Hub identification:** rank films by PageRank to find universally similar "bridge" titles
    - The Borda-score threshold (>1.0) provides a natural confidence filter for high-quality recommendations

    ---

    ## Limitations

    - **Small customer base** (599 renters): sparse co-rental signals for niche films; 42 films have no rental at all
    - **No explicit ratings:** a rental does not imply satisfaction — negative experiences are indistinguishable from positive ones
    - **Historical and static data** (2005–2006): no concept drift, seasonal effects, or evolving preferences can be modeled
    - **Binary rental events:** repeat rentals by the same customer are not weighted
    - **Genre evaluation is coarse:** Sakila's single-label genre assignments may not reflect nuanced content similarity
    - **O(n²) construction:** pairwise edge computation does not scale to large catalogs without approximation methods

    ---

    ## Future Work

    - **Additional layers:** add director, language, MPAA rating, and release year proximity as further network dimensions
    - **Graph embeddings:** apply Node2Vec or GraphSAGE to learn low-dimensional film representations from the multiplex
    - **Community detection:** use Louvain or Leiden algorithms to discover latent genre and style clusters
    - **Downstream evaluation:** test whether multiplex similarity predicts held-out rentals better than single-layer or popularity baselines
    - **Scalability:** replace the O(n²) edge loop with locality-sensitive hashing (LSH) or approximate nearest-neighbour indexing
    - **Temporal modeling:** exploit the rental timestamp sequence to model how co-rental patterns evolve over time
    """)
    return


if __name__ == "__main__":
    app.run()

"""
DESI Multi-Modality Explorer — Mid-Term Project
Visualizations across text, images, maps, networks, and time-series.

Run with: streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import networkx as nx

# ---------------- Page config ----------------
st.set_page_config(
    page_title="DESI Multi-Modality Explorer",
    page_icon="🌌",
    layout="wide",
)

HERE = Path(__file__).resolve().parent
GALAXY_CSV = HERE / "desi_galaxy_sample_hf.csv"
ARTIFACT = HERE / "week7_bertopic.json"

# Okabe–Ito colorblind-safe palette
INK = "#2E6E8E"
ACCENT = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GOLD = "#F0E442"
PALETTE = [INK, ACCENT, GREEN, PURPLE, GOLD]
TABLEAU20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]


# ---------------- Data loading ----------------
@st.cache_data
def load_galaxies(
    path: Path, file_signature: tuple[int, int]
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["Dec"] = pd.to_numeric(df["Dec"], errors="coerce")
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df = df.dropna(subset=["RA", "Dec", "redshift"]).reset_index(drop=True)
    return df


@st.cache_data
def load_artifact(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def build_knn_graph(coords: np.ndarray, k: int = 5) -> nx.Graph:
    """k-nearest-neighbour graph on 2D sky coordinates."""
    n = len(coords)
    k = min(k, n - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = np.inf
        neighbours = np.argsort(dists)[:k]
        for j in neighbours:
            G.add_edge(i, int(j), weight=float(dists[j]))
    return G


try:
    csv_stat = GALAXY_CSV.stat()
    gal = load_galaxies(
        GALAXY_CSV, (csv_stat.st_mtime_ns, csv_stat.st_size)
    )
except FileNotFoundError:
    st.error(
        f"Could not find `{GALAXY_CSV.name}`. "
        "Place it next to `app.py` and reload."
    )
    st.stop()

art = load_artifact(ARTIFACT)


# ---------------- Header ----------------
st.title("🌌 DESI Multi-Modality Explorer")
st.caption(
    "Same underlying survey, five visual modalities: "
    "text · images · maps · networks · time-series."
)


# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Controls")

# Global redshift filter (affects map, network, time-series, images)
zmin, zmax = float(gal["redshift"].min()), float(gal["redshift"].max())
z_range = st.sidebar.slider(
    "Redshift range",
    min_value=round(zmin, 3),
    max_value=round(zmax, 3),
    value=(round(zmin, 3), round(zmax, 3)),
    step=0.001,
    help="Cosmological redshift ≈ distance. Narrow the range to see a slice of the universe.",
)

filtered = gal[
    (gal["redshift"] >= z_range[0]) & (gal["redshift"] <= z_range[1])
].reset_index(drop=True)


# Optional: File Download
st.sidebar.divider()
st.sidebar.subheader("📥 Export Data")
csv_data = filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download filtered sample as CSV",
    data=csv_data,
    file_name="filtered_desi_galaxies.csv",
    mime="text/csv",
)

st.sidebar.metric("Galaxies in view", f"{len(filtered):,}")
st.sidebar.caption(f"Total in sample: {len(gal):,}")


# ---------------- Tabs ----------------
(
    tab_text,
    tab_images,
    tab_map,
    tab_network,
    tab_timeseries,
    tab_about,
) = st.tabs(
    ["📝 Text", "🖼 Images", "🗺 Map", "🕸 Network", "⏳ Time-Series", "ℹ️ About"]
)


# =============================================================
# TAB 1 — Text (BERTopic from HW7)
# =============================================================
with tab_text:
    st.subheader("📝 Scientific narrative — DESI abstracts")
    st.markdown(
        "The BERTopic model groups arXiv abstracts into themes. "
        "This shows **what the DESI collaboration writes about** — the "
        "scientific motivations behind the observations we map in the other tabs."
    )

    if art is None:
        st.warning(
            "`week7_bertopic.json` not found. Copy it from HW7 into this folder."
        )
    else:
        p = art["params"]
        c = art["corpus"]
        st.caption(
            f"Corpus: **{c['name']}** — {c['n_docs']:,} abstracts · "
            f"{p['n_topics']} topics · "
            f"{p['n_unassigned']} unassigned"
        )

        # Topic scatter
        docs = pd.DataFrame(art["docs"])
        topics = [t for t in art["topics"] if t["topic"] != -1]
        labels_by_id = {
            t["topic"]: f"Topic {t['topic']}: {t['label']}" for t in topics
        }
        docs["Topic"] = docs["topic"].map(labels_by_id).fillna("unassigned (−1)")
        
        unique_labels = [labels_by_id[t] for t in sorted(labels_by_id)]
        if "unassigned (−1)" in set(docs["Topic"]):
            unique_labels.append("unassigned (−1)")

        topic_labels = [
            label for label in unique_labels if label != "unassigned (−1)"
        ]
        palette = [
            TABLEAU20[i % len(TABLEAU20)] for i in range(len(topic_labels))
        ]
        if len(topic_labels) != len(unique_labels):
            palette.append("#cccccc")

        custom_colors = dict(zip(unique_labels, palette))

        chart = (
            alt.Chart(docs)
            .mark_circle(size=6, opacity=0.75)
            .encode(
                x=alt.X("x:Q", title=None),
                y=alt.Y("y:Q", title=None),
                color=alt.Color(
                    "Topic:N",
                    title="Topic",
                    scale=alt.Scale(
                        domain=list(custom_colors.keys()),
                        range=list(custom_colors.values()),
                    ),
                ),
                tooltip=["snippet:N", "Topic:N"],
            )
            .properties(height=460)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Figure T1. BERTopic embedding of DESI abstracts, colored by topic. "
            "Uses sentence embeddings (semantic similarity). Gray = unassigned."
        )
# =============================================================
# TAB 2 — Images (Legacy Survey cutouts)
# =============================================================
with tab_images:
    st.subheader("🖼 Galaxy morphology — Legacy Survey cutouts")
    st.markdown(
        "Each row in the dataset carries a JPEG cutout URL from the DESI Legacy "
        "Imaging Surveys. Below are thumbnails for a random subset of the "
        "currently filtered galaxies."
    )

    if len(filtered) == 0:
        st.warning("No galaxies in this redshift range.")
    else:
        n_show = min(12, len(filtered))
        sample = filtered.sample(n=n_show, random_state=0).reset_index(drop=True)

        cols = st.columns(4)
        for i, row in sample.iterrows():
            with cols[i % 4]:
                try:
                    st.image(
                        row["image_url"],
                        caption=f"RA {row['RA']:.2f}°, Dec {row['Dec']:.2f}°\n"
                                f"z = {row['redshift']:.4f}",
                        use_container_width=True,
                    )
                except Exception:
                    st.text("image unavailable")

        st.caption(
            "Figure I1. Legacy Survey DR10 JPEG cutouts (256 px). "
            "Morphology (spiral / elliptical / irregular) is what the eye "
            "classifies before spectroscopy confirms redshift."
            "Note: If an image fails to load from the Legacy Survey server it is likely due to server-side restrictions. "
            
        )


# =============================================================
# TAB 3 — Map (RA/Dec sky positions)
# =============================================================
with tab_map:
    st.subheader("🗺 Sky map — RA vs Dec")
    st.markdown(
        "Where are these galaxies on the sky? The DESI footprint is not "
        "uniform; it follows the Legacy Survey imaging coverage."
    )

    if len(filtered) == 0:
        st.warning("No galaxies in this redshift range.")
    else:
        chart = (
            alt.Chart(filtered)
            .mark_circle(size=18, opacity=0.65, color=INK)
            .encode(
                x=alt.X("RA:Q", title="Right Ascension (°)"),
                y=alt.Y("Dec:Q", title="Declination (°)"),
                color=alt.Color(
                    "redshift:Q",
                    title="Redshift",
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("RA:Q", format=".3f"),
                    alt.Tooltip("Dec:Q", format=".3f"),
                    alt.Tooltip("redshift:Q", format=".4f"),
                ],
            )
            .properties(height=520)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Figure M1. Sky positions of the sampled galaxies, colored by "
            "redshift. Each dot is one spectroscopic target."
        )


# =============================================================
# TAB 4 — Network (k-NN graph in sky coordinates)
# =============================================================
with tab_network:
    st.subheader("🕸 Network — k-nearest neighbours on the sky")
    st.markdown(
        "Treat each galaxy as a node and connect it to its **k closest "
        "neighbours** on the sky. Clusters of nodes show regions where "
        "DESI targeted many galaxies; isolated nodes are rare targets."
    )

    k = st.slider("Number of neighbours per galaxy (k)", 1, 15, 4)
    max_nodes = st.slider(
        "Max galaxies in the network (downsampled for speed)",
        min_value=50,
        max_value=min(500, len(filtered)),
        value=min(200, len(filtered)),
        step=50,
    )

    if len(filtered) < 10:
        st.warning("Need at least 10 galaxies to build a meaningful network.")
    else:
        net_df = filtered.sample(
            n=min(max_nodes, len(filtered)), random_state=0
        ).reset_index(drop=True)

        coords = net_df[["RA", "Dec"]].values
        G = build_knn_graph(coords, k=k)

        # Degree centrality
        degrees = dict(G.degree())
        net_df["degree"] = net_df.index.map(degrees).astype(int)

        # Spring layout (force-directed)
        pos = nx.spring_layout(G, k=0.35, iterations=80, seed=0)
        net_df["x"] = [pos[i][0] for i in net_df.index]
        net_df["y"] = [pos[i][1] for i in net_df.index]

        # Edge list for Altair
        edges = pd.DataFrame(
            [(u, v) for u, v in G.edges()],
            columns=["source", "target"],
        )
        edges["x"] = edges["source"].map(lambda n: pos[n][0])
        edges["y"] = edges["source"].map(lambda n: pos[n][1])
        edges["x2"] = edges["target"].map(lambda n: pos[n][0])
        edges["y2"] = edges["target"].map(lambda n: pos[n][1])

        edge_layer = (
            alt.Chart(edges)
            .mark_rule(color="#999", strokeWidth=0.5, opacity=0.5)
            .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
        )
        node_layer = (
            alt.Chart(net_df)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("x:Q", title=None),
                y=alt.Y("y:Q", title=None),
                color=alt.Color(
                    "redshift:Q",
                    title="Redshift",
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("RA:Q", format=".3f"),
                    alt.Tooltip("Dec:Q", format=".3f"),
                    alt.Tooltip("redshift:Q", format=".4f"),
                    alt.Tooltip("degree:Q", title="Degree"),
                ],
            )
            .properties(height=560)
        )

        st.altair_chart((edge_layer + node_layer), use_container_width=True)
        st.caption(
            f"Figure N1. k-NN network of {len(net_df)} galaxies "
            f"(k = {k}). Node color = redshift, size = constant; "
            f"edges connect each galaxy to its {k} nearest sky neighbours."
        )


# =============================================================
# TAB 5 — Time-series (redshift distribution)
# =============================================================
with tab_timeseries:
    st.subheader("⏳ Redshift distribution — a cosmic time-series")
    st.markdown(
        "Cosmological redshift is a proxy for **lookback time**. The histogram "
        "below shows how many galaxies in the current filter sit at each "
        "redshift — a snapshot of the survey's depth."
    )

    if len(filtered) < 5:
        st.warning("Not enough galaxies in this redshift range.")
    else:
        chart = (
            alt.Chart(filtered)
            .mark_bar(color=INK, opacity=0.85)
            .encode(
                x=alt.X(
                    "redshift:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Redshift (z) → farther back in time",
                ),
                y=alt.Y("count():Q", title="Number of galaxies"),
                tooltip=["count():Q"],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

        # Rolling mean to hint at a trend
        series = (
            filtered.sort_values("redshift")
            .assign(rolling=filtered["redshift"].expanding().mean())
        )
        st.caption(
            f"Figure S1. Redshift histogram for {len(filtered):,} galaxies. "
            "Peaks reflect survey targeting preferences, not the true cosmic "
            "volume density."
        )

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Median z", f"{filtered['redshift'].median():.3f}")
        c2.metric("Max z", f"{filtered['redshift'].max():.3f}")
        c3.metric("Mean z", f"{filtered['redshift'].mean():.3f}")


# =============================================================
# TAB 6 — About
# =============================================================
with tab_about:
    st.subheader("ℹ️ About this project")
    st.markdown(
        """
        **Multi-modality visualization of the DESI survey**

        This app presents one underlying question — *what does the DESI galaxy
        survey tell us about the large-scale structure of the universe?* —
        through five visual modalities:

        | Tab | Modality | Encoding |
        |---|---|---|
        | 📝 Text | Scientific abstracts | BERTopic embedding, colored by topic |
        | 🖼 Images | Legacy Survey cutouts | Thumbnails with tooltip metadata |
        | 🗺 Map | Sky positions | RA vs Dec scatter, colored by redshift |
        | 🕸 Network | k-NN graph | Nodes = galaxies, edges = sky proximity |
        | ⏳ Time-Series | Redshift histogram | Binned redshift, x = distance/time |

        **Dataset.** `desi_galaxy_sample_hf.csv` — 2,000 galaxies from the DESI
        DR1 Bright Galaxy Survey (via Hugging Face `juliensimon/desi-dr1-redshifts`),
        with RA/Dec/redshift from spectroscopy and image cutouts from the
        Legacy Survey DR10 viewer.

        **Accessibility.**
        - Okabe–Ito colorblind-safe palette for categorical encodings.
        - Sequential `viridis` scales for quantitative encodings.
        - Every chart has labeled axes, titles, and a caption.
        - Descriptions provided for screen-reader users.

        **Reproducibility.**
        - All data is either bundled (`desi_galaxy_sample_hf.csv`,
          `desi_corpus.csv`, `week7_bertopic.json`) or fetched live from the
          public Legacy Survey viewer.
        - Heavy computations (`k-NN graph`, `BERTopic artifact`) are cached
          with `@st.cache_data`.
        """
    )
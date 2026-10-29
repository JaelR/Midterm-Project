# DESI Multi-Modality Explorer — Mid-Term Project

## Live App
[Insert your Streamlit Cloud URL here]

## Dataset
The app uses a sample of 2,000 galaxies from the DESI DR1 Bright Galaxy Survey (`desi_galaxy_sample_hf.csv`). 
It also uses the DESI arXiv abstracts corpus (`desi_corpus.csv`) and a pre-computed BERTopic model (`week7_bertopic.json`) to explore the scientific narrative behind the survey.

## Visual Modalities
This project fulfills the mid-term requirement by visualizing the same underlying survey through 5 distinct complex modalities:

1. **📝 Text:** BERTopic semantic embedding of DESI scientific abstracts.
2. **🖼 Images & Spectra:** Legacy Survey image cutouts and dynamic SDSS spectral overlays.
3. **🗺 Maps:** RA/Dec sky distribution of the galaxies.
4. **🕸 Networks:** k-Nearest Neighbor graph connecting galaxies by sky proximity.
5. **⏳ Time-Series:** Redshift (lookback time) distribution histogram.

## Features & Accessibility
- **Interactive Layout:** Multi-tab design with a persistent sidebar for filtering by redshift.
- **Accessibility:** Colorblind-safe palettes (Okabe–Ito, Tableau20, Viridis), redundant markers (shapes + colors), labeled axes, and descriptive alt-text captions.
- **Reproducibility:** All data is included in the repo or fetched from public APIs. Heavy computations (k-NN graph) are cached via `@st.cache_data`.

## Run Locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
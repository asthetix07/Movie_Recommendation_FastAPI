# 🎬 Movie Recommendation System

A full-stack, AI-powered movie recommendation web application that combines a custom content-based machine learning model with real-time data from the TMDB (The Movie Database) API. Built with **FastAPI** on the backend and **Streamlit** on the frontend, the system delivers personalized movie suggestions, live search, and rich movie details — all in a clean, responsive UI.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Dataset](#-dataset)

---

## 🧠 Overview

This project is a complete end-to-end movie recommendation engine. It processes a dataset of ~45,000 movies, builds a TF-IDF based similarity model from each movie's overview, genres, and tagline, and exposes it through a REST API. A Streamlit frontend lets users browse trending movies, search by keyword, view full movie details, and get smart recommendations — all backed by live TMDB data for up-to-date posters, ratings, and metadata.

---

## ✨ Key Features

- **Home Feed** — Browse movies by category: Trending, Popular, Top Rated, Now Playing, or Upcoming. All data is fetched live from TMDB.
- **Smart Search** — Type any keyword and get real-time autocomplete suggestions powered by TMDB's search API. Results are shown as a scrollable poster grid.
- **Movie Details Page** — Click any movie to see its full details: poster, backdrop image, release date, genres, and plot overview.
- **Content-Based Recommendations (TF-IDF)** — For any selected movie, the system computes cosine similarity against a TF-IDF matrix built from ~45,000 movies and returns the most similar titles, enriched with TMDB posters.
- **Genre-Based Recommendations** — Uses TMDB's Discover API to surface popular movies from the same genre.
- **Fully Async Backend** — All TMDB API calls are non-blocking using `httpx` async client, keeping the API fast and efficient.
- **CORS Enabled** — The FastAPI backend is configured with CORS middleware to work smoothly with any frontend client.

---

## ⚙️ How It Works

### 1. Data Preprocessing (Jupyter Notebook)

The `movies.ipynb` notebook handles all the offline data preparation:

1. Loads `movies_metadata.csv` (~45,000 movies from Kaggle).
2. Selects the relevant columns: `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`.
3. Parses the genres column from JSON strings into plain text.
4. Combines `overview`, `genres`, and `tagline` into a single `tags` feature.
5. Applies NLP preprocessing using NLTK:
   - Lowercasing
   - Punctuation removal
   - Stop-word removal
   - Word lemmatization
6. Builds a **TF-IDF matrix** using scikit-learn with `max_features=50,000` and `ngram_range=(1, 2)`.
7. Saves all artifacts as pickle files: `df.pkl`, `indices.pkl`, `tfidf_matrix.pkl`, `tfidf.pkl`.

### 2. FastAPI Backend (`main.py`)

On startup, the API loads the four pickle files into memory and builds a normalized title-to-index map. When a recommendation request comes in:

- It looks up the queried title in the local index.
- Computes cosine similarity between the movie's TF-IDF vector and all other movies.
- Returns the top-N most similar titles.
- Optionally enriches each result with a TMDB poster and metadata using async HTTP calls.

### 3. Streamlit Frontend (`app.py`)

The frontend is a single-file Streamlit app with two views:

- **Home View** — shows a poster grid of movies from the selected home feed category, with a live search bar at the top.
- **Details View** — shows full movie info plus two recommendation sections: TF-IDF similar movies and genre-based suggestions.

Navigation is handled via Streamlit session state and URL query parameters (`?view=details&id=<tmdb_id>`), making it easy to share deep links.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend Framework | FastAPI |
| ASGI Server | Uvicorn |
| Frontend | Streamlit |
| ML / NLP | scikit-learn (TF-IDF), NLTK |
| Data Processing | pandas, NumPy, SciPy |
| HTTP Client (async) | httpx |
| External Data | TMDB API v3 |
| Language | Python 3.11 |
| Config Management | python-dotenv |
| Deployment | Render |

---

## 📁 Project Structure

```
Movie_Recommendation_FastAPI/
│
├── main.py               # FastAPI application — all routes and ML logic
├── app.py                # Streamlit frontend
├── movies.ipynb          # Data preprocessing + TF-IDF model training notebook
│
├── movies_metadata.csv   # Raw dataset (~45,000 movies from Kaggle)
├── df.pkl                # Preprocessed DataFrame (pickled)
├── indices.pkl           # Title-to-index mapping (pickled)
├── tfidf_matrix.pkl      # Trained TF-IDF matrix (pickled, scipy sparse)
├── tfidf.pkl             # Fitted TF-IDF vectorizer (pickled)
│
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version for deployment
└── .python-version       # Python version for local development
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/home` | Home feed — trending, popular, top_rated, now_playing, upcoming |
| `GET` | `/tmdb/search` | Keyword search — returns raw TMDB results with multiple matches |
| `GET` | `/movie/id/{tmdb_id}` | Full details for a specific movie by TMDB ID |
| `GET` | `/recommend/tfidf` | TF-IDF content-based recommendations by movie title |
| `GET` | `/recommend/genre` | Genre-based recommendations by TMDB movie ID |
| `GET` | `/movie/search` | Bundle endpoint — details + TF-IDF recs + genre recs in one call |

Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 1. Clone the repository

```bash
git clone https://github.com/asthetix07/Movie_Recommendation_FastAPI.git
cd Movie_Recommendation_FastAPI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

### 4. Run the FastAPI backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 5. Run the Streamlit frontend

In a separate terminal:

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

> **Note:** If you are running locally, update the `API_BASE` variable in `app.py` from the Render URL to `http://127.0.0.1:8000`.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `TMDB_API_KEY` | Your TMDB API v3 key. Required for all live movie data. Get it free at [themoviedb.org](https://www.themoviedb.org/settings/api). |

---

## 📊 Dataset

- **Source:** [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- **File:** `movies_metadata.csv`
- **Size:** ~45,000 movies
- **Key columns used:** `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`


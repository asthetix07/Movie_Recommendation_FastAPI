<p align="center">
  <h1 align="center">🎬 Cinewatch</h1>
  <p align="center">
    A full-stack movie recommendation engine powered by <strong>FastAPI</strong>, <strong>Streamlit</strong>, and <strong>TF-IDF</strong> content-based filtering — enriched with live data from the <strong>TMDB API</strong>.
  </p>
  <p align="center">
    <a href="https://cinewatch-vercel.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/></a>
    <a href="#features"><img src="https://img.shields.io/badge/Features-✨-blueviolet?style=for-the-badge" alt="Features"/></a>
    <a href="#tech-stack"><img src="https://img.shields.io/badge/Stack-FastAPI%20%2B%20Streamlit-009688?style=for-the-badge" alt="Tech Stack"/></a>
    <a href="#docker"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  </p>
  <p align="center">
    🔗 <strong>Live Demo:</strong> <a href="https://cinewatch-vercel.streamlit.app/">CineMatch — AI Movie Recommendations · Streamlit</a>
  </p>
</p>

---

## 📖 Overview

This project is a **content-based movie recommendation system** that combines:

- **TF-IDF (Term Frequency–Inverse Document Frequency)** cosine similarity on a local movie dataset for intelligent recommendations.
- **TMDB API** integration for real-time movie metadata, posters, trending feeds, and genre-based discovery.
- A **FastAPI** backend serving RESTful endpoints.
- A **Streamlit** frontend providing an interactive, poster-rich UI.

> Search for any movie → get detailed info → receive personalized recommendations powered by both content similarity and genre matching.

> 🚀 **Try it now:** [CineMatch — AI Movie Recommendations · Streamlit](https://cinewatch-vercel.streamlit.app/)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Smart Search** | Keyword-based movie search with autocomplete suggestions via TMDB |
| 🧠 **TF-IDF Recommendations** | Content-based similarity recommendations using cosine similarity on TF-IDF vectors |
| 🎭 **Genre Recommendations** | Discover popular movies in the same genre via TMDB's discover API |
| 🏠 **Home Feed** | Browse trending, popular, top-rated, upcoming, and now-playing movies |
| 📄 **Movie Details** | Full movie details including overview, genres, release date, poster & backdrop |
| ⚡ **Async API** | Fully asynchronous FastAPI backend with `httpx` for non-blocking TMDB calls |

---

## 🏗️ Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│                      │  HTTP   │                      │
│   Streamlit UI       │────────▶│   FastAPI Backend     │
│   (app.py :8501)     │◀────────│   (main.py :8000)     │
│                      │         │                      │
└──────────────────────┘         └────────┬─────────────┘
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                        ┌──────────┐ ┌─────────┐ ┌────────┐
                        │ TMDB API │ │ TF-IDF  │ │ Pickle │
                        │ (live)   │ │ Matrix  │ │ Data   │
                        └──────────┘ └─────────┘ └────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **ML/NLP** | [scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer + Cosine Similarity) |
| **Data** | [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) + [SciPy](https://scipy.org/) |
| **HTTP Client** | [httpx](https://www.python-httpx.org/) (async) |
| **External API** | [TMDB API v3](https://developer.themoviedb.org/docs) |
| **Containerization** | [Docker](https://www.docker.com/) + [Supervisor](http://supervisord.org/) |
| **Language** | Python 3.11+ |

---

## 📂 Project Structure

```
Cinewatch/
├── main.py               # FastAPI backend — API routes & TF-IDF logic
├── app.py                # Streamlit frontend — interactive UI
├── movies.ipynb          # Jupyter notebook — data preprocessing & model training
├── requirements.txt      # Python dependencies
├── Dockerfile            # Multi-service Docker image
├── supervisord.conf      # Supervisor config (runs FastAPI + Streamlit)
├── .env                  # Environment variables (TMDB_API_KEY)
├── .gitignore            # Git ignore rules
├── .dockerignore         # Docker build ignore rules
├── df.pkl                # Preprocessed movie DataFrame
├── indices.pkl           # Title-to-index mapping
├── tfidf.pkl             # Fitted TF-IDF Vectorizer
└── tfidf_matrix.pkl      # Precomputed TF-IDF sparse matrix
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **TMDB API Key** — Get one for free at [themoviedb.org](https://www.themoviedb.org/settings/api)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Cinewatch.git
cd Cinewatch
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 5. Run the Application

**Start the FastAPI backend:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Streamlit frontend** (in a separate terminal):

```bash
streamlit run app.py --server.port 8501
```

> The Streamlit app will connect to the FastAPI backend at `http://localhost:8000` by default. You can override this with the `API_BASE` environment variable.

---

## 🐳 Docker

### Build & Run with Docker

```bash
# Build the image
docker build -t cinewatch .

# Run the container
docker run -d \
  --name cinewatch \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env \
  cinewatch
```

| Service | URL |
|---|---|
| FastAPI (API + Docs) | `http://localhost:8000/docs` |
| Streamlit UI | `http://localhost:8501` |

> The Docker image uses **Supervisor** to run both Uvicorn and Streamlit as managed processes within a single container.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/home?category=popular&limit=24` | Home feed (trending, popular, top_rated, upcoming, now_playing) |
| `GET` | `/tmdb/search?query=batman` | Search movies via TMDB (returns raw results) |
| `GET` | `/movie/id/{tmdb_id}` | Get full movie details by TMDB ID |
| `GET` | `/recommend/genre?tmdb_id=123&limit=18` | Genre-based recommendations |
| `GET` | `/recommend/tfidf?title=Avatar&top_n=10` | TF-IDF content-based recommendations |
| `GET` | `/movie/search?query=inception` | **Bundle** — Details + TF-IDF recs + Genre recs |

> Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the FastAPI server is running.

---

## 🧠 How the Recommendation Engine Works

1. **Data Preprocessing** (`movies.ipynb`)
   - Movie metadata is cleaned, tokenized, and combined into a text "soup" feature.
   - A **TF-IDF Vectorizer** is fitted on the corpus to produce a sparse term-document matrix.
   - The resulting artifacts (`df.pkl`, `indices.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl`) are serialized with `pickle`.

2. **Content-Based Filtering** (`main.py`)
   - When a user selects a movie, the system retrieves its TF-IDF vector.
   - **Cosine similarity** is computed against all other movie vectors in the matrix.
   - The top-N most similar movies are returned as recommendations.

3. **Genre Enrichment** (TMDB API)
   - The system also fetches the movie's genres from TMDB.
   - It uses TMDB's **Discover API** to find popular movies in the same genre, providing a complementary set of recommendations.

---

## 📸 Screenshots

<!-- Add your screenshots here -->
<!-- ![Home Page](screenshots/home.png) -->
<!-- ![Movie Details](screenshots/details.png) -->
<!-- ![Recommendations](screenshots/recommendations.png) -->

*Screenshots coming soon!*

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) — Movie data and posters
- [FastAPI](https://fastapi.tiangolo.com/) — Modern, fast web framework for building APIs
- [Streamlit](https://streamlit.io/) — The fastest way to build data apps
- [scikit-learn](https://scikit-learn.org/) — Machine learning library for TF-IDF vectorization

---

<p align="center">
  Made with ❤️ using FastAPI & Streamlit
</p>

# 🎥 Hybrid Movie Recommendation System with Scikit-learn

This project implements a **content-based recommendation system** using **Scikit-learn** and **Streamlit**. It suggests movies based on genre similarity using TF-IDF and cosine similarity.

## 🔍 Approach

- **Feature Extraction:** TF-IDF on genres column
- **Similarity Measure:** Cosine similarity
- **Framework:** Streamlit dashboard for real-time movie recommendations

## 🧠 Models Used

- `TfidfVectorizer` for feature extraction
- `cosine_similarity` for similarity scores
- (Extendable with collaborative filtering, user-based KNN, or matrix factorization)

## 📁 Dataset

Basic movie metadata (a sample set is provided):

```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
...
```

## 🚀 How to Run

1. Install requirements:

```bash
pip install pandas numpy scikit-learn streamlit
```

2. Run the app:

```bash
streamlit run app/main.py
```

3. Select a movie from the dropdown to get real-time recommendations.

## 💡 Next Steps

- Add user ratings for collaborative filtering
- Integrate with TMDB API for posters and trailers
- Deploy on a Flask/Streamlit-based movie website

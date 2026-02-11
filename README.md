# Recommender System using Matrix Factorization

A production-oriented collaborative filtering system built using the Yelp dataset.  
This project implements Matrix Factorization **from scratch** and also using an optimized library (`cmfrec`) to demonstrate both theoretical depth and practical scalability.

---

## Project Objective

To build a personalized recommendation engine that:

- Learns latent user and business preferences
- Predicts unseen ratings
- Generates Top-K recommendations
- Evaluates ranking quality using Precision@K and Recall@K
- Handles real-world challenges like sparsity and cold start

This project reflects how modern recommendation systems (Swiggy, Zomato, Netflix, Amazon) operate under the hood.

---

## Repository Structure

├── Recommender_System_MF.ipynb # Matrix Factorization from scratch

├── Recommender_system_cmfrec.ipynb # Optimized MF using cmfrec


---

## Dataset

**Source:** Yelp Academic Dataset  

**Files Used:**
- `review.json`
- `business.json`

**Key Columns:**
- `user_id`
- `business_id`
- `stars`

The dataset is highly sparse, simulating real-world recommendation scenarios.

---

## Implementation 1: Matrix Factorization (From Scratch)

Notebook: `Recommender_System_MF.ipynb`

### Steps Covered

- Data ingestion from JSON
- ID encoding for users and businesses
- Train-test split (preventing data leakage)
- Construction of interaction matrix
- Matrix Factorization using Gradient Descent
- Regularization to prevent overfitting
- Full prediction matrix computation
- Evaluation using ranking metrics

### Mathematical Formulation

We approximate the rating matrix:

```
R ≈ P × Qᵀ
```

Where:
- `P` → User latent matrix
- `Q` → Business latent matrix
- `K` → Number of latent dimensions

Each user and business is represented in a shared latent space capturing hidden behavioral patterns.

---

## Implementation 2: Optimized MF using cmfrec

Notebook: `Recommender_system_cmfrec.ipynb`

This implementation demonstrates:

- Efficient large-scale training
- Faster convergence
- Production-grade matrix factorization
- Better scalability for real-world deployment

This mirrors how recommender systems are implemented in industry environments.

---

## Evaluation Metrics

Instead of relying only on regression error, this system evaluates ranking performance:

- RMSE
- Precision@K
- Recall@K

This ensures the model optimizes for recommendation relevance, not just numerical accuracy.

---

## Engineering Considerations

- Sparse data handling
- Hyperparameter tuning (K, learning rate, regularization)
- Vectorized operations for performance
- Proper encoding to maintain ID consistency
- Cold-start strategy discussion (global average, bias models)

---

## Key Skills Demonstrated

- Collaborative Filtering
- Matrix Factorization (Gradient Descent)
- Ranking Metric Evaluation
- Data Leakage Prevention
- Sparse Data Optimization
- Model Evaluation Strategy
- Production-Oriented Thinking

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- cmfrec
- Google Colab

---

## How to Run

1. Clone the repository
2. Download the Yelp dataset
3. Update dataset paths inside the notebooks
4. Run notebooks sequentially

---

## Future Improvements

- Implement Alternating Least Squares (ALS)
- Add implicit feedback modeling
- Deploy model via FastAPI
- Add MLflow model tracking
- Scale using sparse matrix frameworks

---

## Why This Project Stands Out

- Implements Matrix Factorization from scratch (not just library usage)
- Uses proper ranking metrics (Precision@K, Recall@K)
- Addresses real-world challenges like sparsity and cold start
- Compares theoretical and optimized implementations

This project demonstrates strong fundamentals in recommender systems and applied machine learning.

---

## Author

Aasim  
Focused on building scalable ML systems with a strong foundation in recommendation algorithms and behavioral data modeling.

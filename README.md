# IMDb Movie Rating Predictor  

This project explores machine learning techniques to predict IMDb movie ratings using metadata such as **genres, budgets, revenues, overviews, and production details**. By combining **text mining** (TF-IDF on movie overviews) with **feature engineering** and **classification algorithms**, the project evaluates multiple models and identifies the most effective subset of attributes for rating prediction.  

This Project delves into various Machine Learning Algorhtims, such as Random Forest, Complement Naive Bayes,

---

## Dataset  
- IMDb movies dataset (10,178 instances)  
- Features:  
  - **Numerical:** budget, revenue  
  - **Categorical:** genre, production status, country, language  
  - **Textual:** movie overview  
  - **Target:** rating (binned into 5 classes: Very Bad, Bad, Satisfactory, Good, Very Good)  

---

## Preprocessing  
- Removed non-English entries, duplicates, and converted budget/revenue to floats.  
- Normalized budget and revenue to reduce scale skew.  
- Discretized continuous ratings into 5 classes.  
- One-hot encoded categorical features such as genres with `MultiLabelBinarizer`.  
- Processed overview text with tokenization, stopword removal, and lemmatization.  
- Applied **TF-IDF vectorization** to represent overviews numerically.  
- Used **forward feature selection** to identify the most informative attributes: TF-IDF, budget, revenue, and genre.  

---

## Models Evaluated  
The following classifiers were trained and tested with an 80/20 stratified split:  

- Random Forest  
- Complement Naive Bayes (CNB)  
- Logistic Regression  
- Support Vector Machines (SVM)  
- K-Nearest Neighbors (K-NN)  

Each was tested both with and without feature selection.  

---

## Results & Conclusion  

| Model                | Accuracy (No FS) | Accuracy (With FS) | Key Notes |
|-----------------------|------------------|--------------------|-----------|
| Random Forest         | 63%              | **67%**            | Best balance of precision & recall, strong on minority classes |
| Complement Naive Bayes| 62–63%           | 62%                | Good for imbalance, but failed on Class 4 |
| Logistic Regression   | ~60%             | ~60%               | Little change |
| SVM                   | ~61%             | ~61%               | Moderate on rare classes |
| K-NN                  | 64%              | 60%                | Better without FS |

**Random Forest with feature selection** was the most effective:  
- Highest accuracy (**67%**)  
- Major recall gains on minority classes (Class 0 recall improved from 0.19 → 0.86)  
- Perfect precision for Class 0 (1.00), avoiding false positives  

## Conclusion  

This study aimed to predict IMDb movie ratings using Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (K-NN), Complement Naive Bayes (CNB), and Random Forest. The dataset was preprocessed by removing duplicates, normalizing budget and revenue values, applying one-hot encoding to categorical features, and using TF-IDF vectorization for the movie overviews. Each model was tested first on the full attribute set and then again after applying forward feature selection, which identified TF-IDF, budget, revenue, and genre as the most informative features.  

Among all the models, **Random Forest with feature selection** achieved the best overall results. It reached a test accuracy of **67%** and showed substantial improvements in recall for minority classes. For instance, recall for Class 0 (very poorly rated movies) improved from **0.19 → 0.86**, showing the model’s ability to better identify underrepresented cases once the feature space was refined. Random Forest also achieved **perfect precision (1.00)** for Class 0, which ensured that every movie flagged as very poorly rated truly belonged to that category, completely avoiding false positives.  

Other models demonstrated narrower strengths. **Complement Naive Bayes** performed consistently in the 62–63% range and was especially effective at handling class imbalance when feature selection was not applied, although it struggled with some of the higher rating categories. **SVM** and **Logistic Regression** both remained steady at around 60–61% accuracy and showed little difference when feature selection was introduced, limiting their flexibility. **K-NN** performed relatively well without feature selection (64%) but declined in performance when the dimensionality was reduced, indicating that it relied more heavily on the full feature space.  

Overall, Random Forest with feature selection demonstrated the strongest and most balanced performance across all metrics. It provided high accuracy, significant gains in recall for underrepresented classes, and perfect precision in identifying very poorly rated films. 

However, the best choice of algorithm depends on the intended use case. If the goal is to highlight or promote highly rated movies, avoiding false positives becomes critical, and Random Forest with feature selection is the best option. If the goal is to capture as many poorly rated movies as possible for moderation or review, then Complement Naive Bayes or Random Forest with feature selection is preferable, since both achieve much higher recall for poorly rated categories.  

---

## Key Takeaways  

- Use **Random Forest with feature selection** when both precision and recall are important, as it combines the highest overall accuracy (**67%**), major improvements in recall for minority classes (**Class 0 recall from 0.19 → 0.86**), and **perfect precision for Class 0 (1.00)**.  
- Use **Complement Naive Bayes (CNB)** when the priority is maximizing recall for poorly rated movies, as it is particularly effective in identifying low-quality films even if overall accuracy is lower.  

---

## Repository Structure  

---

## Functions & Workflow  

### Preprocessing  
- `clean_text(text)` → Tokenizes, removes stopwords, applies lemmatization  
- `vectorize_text(corpus)` → Applies TF-IDF  
- `encode_genres(data)` → One-hot encodes genres  
- `normalize_numeric(data)` → Normalizes budget/revenue  

### Feature Selection  
- `forward_selection(X, y, model)` → Greedy forward attribute selection  

### Model Training & Evaluation  
- `train_model(model, X_train, y_train)`  Trains model   with the corresponding ML algo
- `evaluate_model(model, X_test, y_test)` → Reports accuracy, precision, recall, F1-score  

## How to Run  
```bash
# Clone repo
git clone https://github.com/yourusername/imdb-rating-predictor.git
cd imdb-rating-predictor

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook IMDBClassiferProj.ipynb

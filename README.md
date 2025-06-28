# Heart Disease Prediction Project

## 1. General Description

This project analyzes, predicts, and visualizes heart disease risks using machine learning. The workflow covers data preprocessing, feature selection, dimensionality reduction (PCA), model training, evaluation, and deployment. Both supervised (Logistic Regression, Decision Trees, Random Forest, SVM) and unsupervised learning (K-Means, Hierarchical Clustering) techniques are used. A Streamlit app is developed for user interaction.

---

## 1.1 Objectives

- Perform data preprocessing & cleaning (handle missing values, encoding, scaling)
- Apply dimensionality reduction (PCA) to retain essential features
- Implement feature selection using statistical and ML-based methods
- Train supervised models: Logistic Regression, Decision Trees, Random Forest, SVM
- Apply unsupervised learning: K-Means, Hierarchical Clustering for pattern discovery
- Optimize models using hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Deploy a Streamlit UI for real-time user interaction 

---

## 1.2 Tools & Technologies

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow/Keras (optional)
- **Dimensionality Reduction & Feature Selection:** PCA, Recursive Feature Elimination (RFE), Chi-Square Test
- **Supervised Learning Models:** Logistic Regression, Decision Trees, Random Forest, SVM
- **Unsupervised Learning Models:** K-Means, Hierarchical Clustering
- **Model Optimization:** GridSearchCV, RandomizedSearchCV
- **Deployment:** Streamlit (UI), GitHub (version control and hosting)

---

## 2. Project Workflow & Steps

### 2.1 Data Preprocessing & Cleaning
- Load Heart Disease UCI dataset.
- Handle missing values via imputation or removal.
- Encode categorical variables (one-hot encoding).
- Standardize numerical features (MinMaxScaler or StandardScaler).
- Perform Exploratory Data Analysis (EDA) with plots (histograms, correlation heatmaps, boxplots).

### 2.2 Dimensionality Reduction (PCA)
- Apply PCA to reduce feature dimensions while retaining variance.
- Choose the optimal number of components using explained variance ratio.
- Visualize PCA results (scatter plot, cumulative variance plot).

### 2.3 Feature Selection
- Rank features using importance from Random Forest / XGBoost.
- Use Recursive Feature Elimination (RFE).
- Perform Chi-Square tests for significance.
- Select the most relevant features for modeling.

### 2.4 Supervised Learning - Classification Models
- Split data into training (80%) and testing (20%).
- Train Logistic Regression, Decision Tree, Random Forest, and SVM.
- Evaluate using accuracy, precision, recall, F1-score, ROC curve, and AUC.

### 2.5 Unsupervised Learning - Clustering
- Apply K-Means with the elbow method to find optimal clusters.
- Perform Hierarchical Clustering and analyze dendrograms.
- Compare clusters with actual disease labels.

### 2.6 Hyperparameter Tuning
- Optimize models using GridSearchCV and RandomizedSearchCV.
- Compare optimized models against baseline performance.

### 2.7 Model Export & Deployment
- Save the final trained model pipeline (preprocessing + model) as `.pkl` file.
- Ensure reproducibility and easy deployment.

### 2.8 Streamlit Web UI Development 
- Build an interactive Streamlit app for user input and prediction.
- Provide real-time prediction and visualization.

---

## 3. Final Deliverables

- Cleaned dataset with selected features
- PCA results and visualizations
- Trained supervised and unsupervised models
- Model performance metrics and evaluation reports
- Hyperparameter optimized models
- Exported model pipeline (`final_model.pkl`)
- GitHub repository with full source and documentation
- Functional Streamlit app for predictions 

---


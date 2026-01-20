# Analysis of Credit Card Fraud Detection Using Sampling Techniques

---

## 1. Problem Statement
Credit card fraud detection involves identifying fraudulent transactions from a large number of genuine transactions. One of the major challenges in this domain is **class imbalance**, where fraudulent cases form only a very small portion of the dataset.  
This project studies how different **sampling methods** influence the performance of various **machine learning classifiers** after balancing the dataset.

---

## 2. Dataset Overview
- Dataset File: CreditCard_data.csv  
- Output Variable: Class  
  - 0 indicates a valid transaction  
  - 1 indicates a fraudulent transaction  

Due to the skewed nature of the dataset, direct model training can lead to biased predictions.

---

## 3. Methodology

### 3.1 Handling Data Imbalance
The dataset is first balanced using **Random Oversampling**, where minority class samples are duplicated until both classes contain an equal number of observations. This step ensures fair learning for all models.

### 3.2 Sampling Approaches
After balancing, multiple sampling approaches are applied independently to examine their effect on model accuracy:

- **Random Sampling**: Selects records randomly from the dataset.
- **Systematic Sampling**: Chooses records at regular intervals.
- **Stratified Sampling**: Maintains class proportions while sampling.
- **Cluster Sampling**: Divides data into groups and selects one group.
- **Bootstrap Sampling**: Generates a new dataset using sampling with replacement.

Each approach produces a separate dataset for experimentation.

### 3.3 Feature Normalization
All numerical features are scaled using **standard normalization** to bring them onto a common scale. This improves convergence and performance for models sensitive to feature magnitude.

### 3.4 Model Selection
The following classification algorithms are trained on each sampled dataset:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- Support Vector Machine  

### 3.5 Evaluation Strategy
- Dataset split: 80% training and 20% testing  
- Sampling is stratified to preserve class distribution  
- Performance metric: **Accuracy**

---

## 4. Experimental Results

### 4.1 Accuracy Comparison Table
The table below presents the accuracy scores (%) achieved by each model across different sampling techniques.

| Model | Bootstrap | Cluster | Random | Stratified | Systematic |
|------|----------|---------|--------|------------|------------|
| DT   | 100.00   | 97.40   | 99.46  | 96.74      | 98.69      |
| LR   | 93.46    | 89.61   | 93.48  | 94.02      | 89.54      |
| NB   | 70.59    | 68.83   | 76.63  | 76.63      | 73.86      |
| RF   | 100.00   | 100.00  | 100.00 | 100.00     | 100.00     |
| SVM  | 98.37    | 98.70   | 98.91  | 97.83      | 98.04      |

---

### 4.2 Result Visualization
The following graph provides a visual comparison of model accuracy under different sampling techniques, making performance trends easier to interpret.

![Accuracy Comparison](images/results.png)

---

### 4.3 Best Performing Sampling Method
The optimal sampling technique for each classifier based on accuracy is summarized below:

| Model | Optimal Sampling Method |
|------|-------------------------|
| Decision Tree | Bootstrap |
| Logistic Regression | Stratified |
| Naive Bayes | Random |
| Random Forest | Bootstrap |
| Support Vector Machine | Random |

---

## 5. Observations and Discussion
- Random Forest consistently performs well regardless of the sampling method, indicating strong generalization.
- Logistic Regression benefits from sampling methods that preserve class proportions.
- Naive Bayes shows comparatively lower performance, suggesting sensitivity to data distribution.
- SVM maintains stable accuracy across most sampling strategies.
- Sampling choice plays a crucial role in improving results for simpler models.

---

## 6. Conclusion
This project demonstrates that applying appropriate sampling techniques significantly improves the effectiveness of fraud detection models.  
While ensemble models remain robust across methods, other classifiers show noticeable performance variation depending on the sampling strategy used.  
Careful preprocessing and sampling selection are therefore essential in real-world fraud detection systems.

---

## 7. Execution Instructions

```bash
pip install pandas numpy scikit-learn imbalanced-learn

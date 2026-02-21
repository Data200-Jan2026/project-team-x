# 📚 Week 2: Literature Review and Dataset Selection

---

## 📂 Dataset Selected

**Bank Marketing Dataset** — UCI Machine Learning Repository (also available on Kaggle)

This is the classic bank marketing dataset originally uploaded to the UCI Machine Learning Repository. It contains information about a direct marketing campaign conducted by a financial institution, where clients were contacted by phone to promote term deposit subscriptions. The dataset includes client demographic details, contact information, and campaign history, with a binary target variable `y` indicating whether the client subscribed (`yes`) or not (`no`).

The purpose of analyzing this dataset is to identify patterns and key factors that influence subscription decisions, which can then inform and improve future marketing campaign strategies for the bank. Its widespread use in academic literature — as confirmed by all three reviewed studies — validates its suitability and credibility for this project.

---

## 📖 Literature Reviews

---

### Review 1: Moro, S., Cortez, P., & Rita, P. (2014)
**A Data-Driven Approach to Predict the Success of Bank Telemarketing**
*Decision Support Systems, Elsevier. DOI: 10.1016/j.dss.2014.03.001*

This study is the original research behind the Bank Marketing Dataset. The authors applied a data mining approach using 150 features related to bank client demographics, product attributes, and social-economic indicators. After semi-automatic feature selection, 22 key features were retained. Four models were compared — Logistic Regression, Decision Tree, Neural Network, and SVM — evaluated using AUC and ALIFT. The Neural Network performed best (AUC = 0.8, ALIFT = 0.7), and sensitivity analysis identified the Euribor rate, call direction, and agent experience as the most influential predictors. This paper provides the foundational methodology and dataset for our project.

---

### Review 2: Jiang, Y. (2018)
**Using Logistic Regression Model to Predict the Success of Bank Telemarketing**
*International Journal on Data Science and Technology, Vol. 4, No. 1, pp. 35–41. DOI: 10.11648/j.ijdst.20180401.15*

This study directly aligns with our chosen methodology — Logistic Regression. Jiang compared LR against Naïve Bayes, SVM, Neural Network, and Decision Tree using the same UCI Bank Marketing Dataset, implemented in R. Results showed that Logistic Regression outperformed the other models in both prediction accuracy and AUC. The study confirms that LR is not only interpretable but also competitive in predictive performance for this type of binary classification problem, supporting our choice of LR as the primary model.

---

### Review 3: Saragih, et al. (2019)
**Long-Term Deposits Prediction: A Comparative Framework of Classification Models for Predicting the Success of Bank Telemarketing**
*Journal of Physics: Conference Series. DOI: 10.1088/1742-6596/1175/1/012035*

This paper provides the broadest model comparison among the three reviews, testing seven algorithms: Decision Tree, Naïve Bayes, Random Forest, K-NN, SVM, Neural Network, and Logistic Regression on the same UCI dataset. Evaluation was based on Accuracy and AUC. SVM yielded the highest results with 97.07% accuracy and AUC of 0.925. This study is useful for benchmarking — it shows what performance levels are achievable, and where Logistic Regression stands relative to more complex models. It also reinforces that simpler, interpretable models like LR remain relevant despite being outperformed on raw accuracy.

---

## 📊 Summary Table

| Study | Models Compared | Best Model | AUC | Key Features Identified |
|---|---|---|---|---|
| Moro et al. (2014) | LR, DT, NN, SVM | Neural Network | 0.80 | Euribor rate, call direction, agent experience |
| Jiang (2018) | LR, NB, SVM, NN, DT | Logistic Regression | Highest among tested | Client demographics, contact type |
| Saragih et al. (2019) | DT, NB, RF, K-NN, SVM, NN, LR | SVM | 0.925 | Multiple campaign attributes |

---

## ✅ Justification for Dataset Selection

All three reviewed studies use the UCI Bank Marketing Dataset, confirming its suitability and credibility for this type of research. The dataset is well-documented, publicly available, and has been validated across multiple comparative studies. It directly supports our research questions around customer subscription behavior and the effectiveness of telemarketing campaigns, making it the ideal choice for this project.
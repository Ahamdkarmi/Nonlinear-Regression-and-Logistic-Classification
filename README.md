# Customer Churn Prediction — Non-Linear & Logistic Regression

##  Overview

This project is part of **Assignment #2** for the *Machine Learning and Data Science (ENCS5341)* course at **Birzeit University**.

It covers two main components:

1. **Non-Linear Regression** — Fitting a noisy sinusoidal signal using Ridge Regression and RBF (Radial Basis Function) networks.
2. **Logistic Regression** — Predicting customer churn using linear and polynomial decision boundaries on a real-world customer dataset.

---

##  Objectives

* Apply ridge regression with multiple regularization values and analyze generalization
* Implement RBF-based non-linear regression with varying numbers of basis functions
* Build and evaluate logistic regression models with linear and non-linear decision boundaries
* Compare model complexity vs. generalization performance
* Select the best model using validation metrics and visualize the ROC curve

---

##  Project Structure

```
.
├── non-linear.ipynb         # Part 1: Ridge Regression & RBF Regression (Jupyter Notebook)
├── Logistic.py              # Part 2: Logistic Regression (linear + polynomial)
├── Customer_data.csv        # Customer churn dataset (from Assignment #1)
├── Representation.pptx      # Presentation summarizing findings
└── README.md
```

---

##  Part 1 — Non-Linear Regression

###  A) Ridge Regression

* Generated 25 data points where `y = sin(5πx) + ε`, with `x ∈ [0,1]` and noise `ε ∈ [-0.3, 0.3]`
* Feature vector: `x⃗ = (1, x, x², ..., x⁹)`
* Applied **Ridge Regression** with 5 values of `λ`: `{0, 0.001, 0.01, 0.1, 1}`
* Plotted fitted curves and analyzed underfitting vs. overfitting

###  B) RBF Regression

* Applied non-linear regression using **RBF basis functions** (no regularization)
* Tested 4 configurations: **1, 5, 10, and 50** RBF centers
* Centers evenly spaced across `[0, 1]`, width chosen empirically
* Compared results against the true function `sin(5πx)`

---

##  Part 2 — Logistic Regression

###  Dataset

The customer churn dataset from Assignment #1, containing:

| Feature        | Description                        |
|----------------|------------------------------------|
| `CustomerID`   | Unique identifier                  |
| `Age`          | Customer age                       |
| `Gender`       | Male / Female                      |
| `Income`       | Annual income                      |
| `Tenure`       | Years with the company             |
| `ProductType`  | Basic / Premium                    |
| `SupportCalls` | Number of support calls            |
| `ChurnStatus`  | Target variable (0 = stayed, 1 = churned) |

###  Preprocessing Steps

* Filled missing values (Median → Age, Income, Tenure | Mode → SupportCalls)
* Handled outliers using **IQR method** (Income → mean replacement, Age → row removal, SupportCalls → median replacement)
* Encoded categorical variables using **Label Encoding**
* Normalized features using **Min-Max Scaling**

###  Dataset Split

| Split      | Size |
|------------|------|
| Training   | 2500 |
| Validation | 500  |
| Test       | 500  |

###  Models Trained

| Model                    | Description                              |
|--------------------------|------------------------------------------|
| Linear Logistic Regression | Standard logistic regression            |
| Polynomial Degree 2      | Non-linear boundary via PolynomialFeatures |
| Polynomial Degree 5      | Higher complexity polynomial boundary    |
| Polynomial Degree 9      | High-degree polynomial (risk of overfitting) |

###  Evaluation Metrics

* Accuracy, Precision, Recall — on training, validation, and test sets
* ROC Curve + AUC — for the best model selected via validation accuracy

---

##  Key Findings

* For Ridge Regression, moderate `λ` values reduced overfitting better than `λ = 0`
* For RBF regression, **10 RBF functions** offered the best balance between underfitting and overfitting
* For Logistic Regression, the linear model and low-degree polynomials generalized best
* High-degree polynomials (degree 9) showed signs of **overfitting** on training data

---

##  How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 2. Run Part 1 (Non-Linear Regression)

Open and run the Jupyter Notebook:

```bash
jupyter notebook non-linear.ipynb
```

### 3. Run Part 2 (Logistic Regression)

```bash
python Logistic.py
```

> Make sure `Customer_data.csv` is in the same directory.

---

##  Visualizations Included

* Ridge regression curves for multiple `λ` values
* RBF regression fits for 1, 5, 10, and 50 basis functions vs. true function
* Confusion matrices for all logistic models
* Classification reports (Accuracy, Precision, Recall)
* ROC curve with AUC for the best model

---

##  References

###  Project Files

* [ Source Code (main.py)](src/main.py)
* [ Dataset (customer_data.csv)](data/customer_data.csv)
* [ Report (Report.pdf)](doc/Report.pdf)

---

###  Official Documentation

* [Scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](https://numpy.org/doc/)
* [Matplotlib](https://matplotlib.org/stable/contents.html)
* [Pandas](https://pandas.pydata.org/docs/)

---

##  Author

* **Ahmad Karmi**
* Course: Machine Learning and Data Science — ENCS5341
* Institution: Birzeit University

---

##  Notes

This project was developed as part of a university machine learning assignment. The dataset is synthetic and used for educational purposes only.

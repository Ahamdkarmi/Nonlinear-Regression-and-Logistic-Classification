import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# ---- Load dataset ----
df = pd.read_csv("Customer_data.csv")

# ---- Fill missing values ----
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
df["SupportCalls"] = df["SupportCalls"].fillna(df["SupportCalls"].mode()[0])

# ---- Fix outliers: Income ----
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1
Upper_Limit = Q3 + 1.5 * IQR
Lower_Limit = Q1 - 1.5 * IQR
Income_mean = df['Income'].mean()
df['Income'] = df['Income'].apply(lambda x: Income_mean if x < Lower_Limit or x > Upper_Limit else x)

# ---- Fix outliers: Age ----
Q1_Age = df['Age'].quantile(0.25)
Q3_Age = df['Age'].quantile(0.75)
IQR_Age = Q3_Age - Q1_Age
Upper_Limit_Age = Q3_Age + 1.5 * IQR_Age
Lower_Limit_Age = Q1_Age - 1.5 * IQR_Age
df = df[(df['Age'] >= Lower_Limit_Age) & (df['Age'] <= Upper_Limit_Age)]

# ---- Fix outliers: SupportCalls ----
Q1_SC = df['SupportCalls'].quantile(0.25)
Q3_SC = df['SupportCalls'].quantile(0.75)
IQR_SC = Q3_SC - Q1_SC
Upper_SC = Q3_SC + 1.5 * IQR_SC
Lower_SC = Q1_SC - 1.5 * IQR_SC
SC_median = df['SupportCalls'].median()
df['SupportCalls'] = df['SupportCalls'].apply(lambda x: SC_median if x < Lower_SC or x > Upper_SC else x)

# ---- Convert categorical columns to numbers ----
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ---- Scale numeric columns ----
numeric_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---- Split dataset ----
X = df.drop(columns=["ChurnStatus"])
y = df["ChurnStatus"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=2500, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=500, shuffle=True, random_state=42)

# ---- Linear Logistic Regression ----
print("\n" + "="*60)
print("   Linear Logistic Regression")
print("="*60)

# ---- Train Linear Logistic Regression ----
linear_model = LogisticRegression(max_iter=1000)
linear_model.fit(X_train, y_train)

# ---- Validate the model ----
val_predictions = linear_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print("\n[Validation Metrics]")
print(f"Accuracy: {val_accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_predictions))
print("Classification Report:")
print(classification_report(y_val, val_predictions, digits=4))

# ---- Evaluate on test dataset ----
test_predictions = linear_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("\n[Test Metrics]")
print(f"Accuracy: {test_accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))
print("Classification Report:")
print(classification_report(y_test, test_predictions, digits=4))
print("="*60 + "\n")

# ---- Non-Linear Logistic Regression ----
degrees = [2, 5, 9]
poly_models = {}
val_accuracies = {"Linear": val_accuracy}
for deg in degrees:
    print("\n" + "=" * 60)
    print(f"   Polynomial Degree: {deg}")
    print("=" * 60)
    # ---- Create pipeline and train ----
    model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    # ---- Validate the model ----
    val_predictions = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_predictions)
    val_accuracies[f"Poly {deg}"] = val_acc
    poly_models[f"Poly {deg}"] = model
    print("\n[Validation Metrics]")
    print(f"Accuracy: {val_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_predictions))
    print("Classification Report:")
    print(classification_report(y_val, val_predictions, digits=4))
    # ---- Evaluate on test dataset ----
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("\n[Test Metrics]")
    print(f"Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))
    print("Classification Report:")
    print(classification_report(y_test, test_predictions, digits=4))
    print("=" * 60 + "\n")

# ---- Select Best Model based on Validation Accuracy ----
all_models = {"Linear": linear_model, **poly_models}
best_model_name = max(val_accuracies, key=val_accuracies.get)
best_model = all_models[best_model_name]
print("\n" + "="*60)
print(f"Best Model based on Validation Accuracy: {best_model_name} ({val_accuracies[best_model_name]:.4f})")
print("="*60)

# ---- ROC Curve and AUC for Best Model ----
if hasattr(best_model, "predict_proba"):
    y_val_probs = best_model.predict_proba(X_val)[:, 1]
else:
    y_val_probs = best_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_probs)
RocCurveDisplay.from_predictions(y_val, y_val_probs)
plt.title(f"ROC Curve - {best_model_name} (AUC = {roc_auc:.4f})")
plt.show()
print(f"AUC for {best_model_name}: {roc_auc:.4f}")

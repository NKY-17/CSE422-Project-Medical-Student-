import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


#  Dataset

df = pd.read_csv('/content/medical_students_dataset.csv')

print(" Dataset Description")
print(f"Initial Dataset Shape: {df.shape} (Data points: {df.shape[0]}, Features: {df.shape[1]})\n")
df.info()

target_col = 'Diabetes'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not available. Found columns: {list(df.columns)}")

# Dropping
df.dropna(how='all', inplace=True)
df.dropna(subset=[target_col], inplace=True)

# Drop non-predictive
if 'Student ID' in df.columns:
    df.drop('Student ID', axis=1, inplace=True)

# Separate features and Target
X_raw = df.drop(target_col, axis=1)
y_raw = df[target_col]

num_cols = X_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

print(f"\nQuantitative Feature ({len(num_cols)}): {num_cols}")
print(f"Categorical Feature ({len(cat_cols)}): {cat_cols}")


# 2.  Exploratory Data Analysis

print("\n Visual EDA ")

#  Imbalanced Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_raw, palette='viridis')
plt.title('Target Class Distribution (Diabetes)', fontweight='bold')
plt.ylabel('Count')
plt.xlabel('Diabetes Status')
plt.tight_layout()
plt.show()

class_counts = y_raw.value_counts()
print(f"Target Class Instance Counts:\n{class_counts}")


if num_cols:
    n_num = len(num_cols)
    n_cols_grid = 3
    n_rows_grid = (n_num + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 4 * n_rows_grid))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=y_raw, y=X_raw[col], ax=axes[i], palette='Set2')
        axes[i].set_title(f'Boxplot: {col} by Diabetes Status', fontweight='bold')
        axes[i].set_xlabel('Diabetes Status')
        axes[i].set_ylabel(col)


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Raw Feature Boxplots (Outlier & Separation Check)', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if cat_cols:
    n_cat = len(cat_cols)
    n_cols_grid = 3
    n_rows_grid = (n_cat + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 4 * n_rows_grid))
    axes = axes.flatten() if n_cat > 1 else [axes]

    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, hue=target_col, ax=axes[i], palette='magma')
        axes[i].set_title(f'Bar Chart: {col} Count by Diabetes', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].legend(title='Diabetes', loc='upper right')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Categorical Feature Distributions by Target Class', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#  Correlation Heatmap

temp_df = X_raw[num_cols].copy()
plt.figure(figsize=(10, 8))
corr_matrix = temp_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap (Quantitative Features)', fontweight='bold')
plt.tight_layout()
plt.show()

print("\nEDA Observation: Correlation matrix indicates collinearity between 'Weight' , 'BMI'.")

# Scatter Matrix
plot_cols = [c for c in ['Age', 'BMI', 'Blood Pressure', 'Cholesterol'] if c in num_cols]
if plot_cols:
    plot_df = X_raw[plot_cols].copy()
    plot_df['Diabetes'] = y_raw
    plot_df_sample = plot_df.sample(n=min(1000, len(plot_df)), random_state=42)

    sns.pairplot(plot_df_sample, hue='Diabetes', palette='viridis', corner=True)
    plt.suptitle('Scatter Matrix of Key Quantitative Features', y=1.02, fontweight='bold')
    plt.show()

# 3. Pre-processing

print("\n Dataset Pre-processing ")

X = X_raw.copy()
y = y_raw.copy()


if 'Weight' in X.columns:
    X.drop('Weight', axis=1, inplace=True)
    if 'Weight' in num_cols:
        num_cols.remove('Weight')
    print("Dropped 'Weight'")

print("\nMissing values count before imputation:")
print(X.isnull().sum())


num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

#  Label Encoding
cat_encoders = {}
for col in cat_cols:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])
    cat_encoders[col] = le_col

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)  # e.g., No -> 0, Yes -> 1

#  Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n shape after  pre-processing: {X_scaled.shape}")


# 4. Dataset Splitting

print("\n Dataset Splitting")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
)
print(f"Train set shape: {X_train.shape} (80%)")
print(f"Test set shape:  {X_test.shape} (20%)")


# 5. Model Training & Testing

print("\nModel Training ")
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42)
}

predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"Training {name}  model ")
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities[name] = model.predict_proba(X_test)[:, 1]
    else:
        probabilities[name] = predictions[name]

# Unsupervised KMeans Clustering
wcss = []
for k in range(1, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
plt.title('Elbow Method', fontweight='bold')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
unique, counts = np.unique(clusters, return_counts=True)
print(f"\nKMeans Clustered Groups: {dict(zip(unique, counts))}")

# KMeans Cluster Scatter Plot
sample_idx = np.random.choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_scaled[sample_idx, X.columns.get_loc('BMI')],
    y=X_scaled[sample_idx, X.columns.get_loc('Blood Pressure')],
    hue=clusters[sample_idx],
    palette='Set1',
    alpha=0.7
)
plt.title('KMeans Clusters Scatter Plot (BMI vs Blood Pressure)', fontweight='bold')
plt.xlabel('BMI (Scaled)')
plt.ylabel('Blood Pressure (Scaled)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


# 6. Model Evaluation &

print("\n Model Eval")

# Sample Predictions Table
results_df = pd.DataFrame({'Actual Label': y_test})
for name in models.keys():
    results_df[f'{name} Pred'] = predictions[name]

results_df['Actual Label'] = target_encoder.inverse_transform(results_df['Actual Label'])
for name in models.keys():
    results_df[f'{name} Pred'] = target_encoder.inverse_transform(results_df[f'{name} Pred'])

print("\nFirst 15 Test Sample Predictions:")
print(results_df.head(15))

acc_scores, prec_scores, rec_scores, f1_scores, cm_dict = {}, {}, {}, {}, {}

for name in models.keys():
    y_pred = predictions[name]
    y_prob = probabilities[name]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    acc_scores[name] = acc
    prec_scores[name] = prec
    rec_scores[name] = rec
    f1_scores[name] = f1
    cm_dict[name] = cm

    print(f"\ {name} MEASURE ")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")

#  Accuracy Compare Graph
plt.figure(figsize=(8, 5))
plt.bar(acc_scores.keys(), acc_scores.values(), color=['#2b5c8f', '#6b5b95', '#feb236', '#d64161'])
plt.title('Prediction Accuracy Across Models', fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1.08)
for i, v in enumerate(acc_scores.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.show()

# Precision, Recall, F1 Score Compare Graph
x_labels = np.arange(len(models))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x_labels - width, prec_scores.values(), width, label='Precision', color='skyblue')
rects2 = ax.bar(x_labels, rec_scores.values(), width, label='Recall', color='salmon')
rects3 = ax.bar(x_labels + width, f1_scores.values(), width, label='F1 Score', color='lightgreen')

ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, F1 Score Compare Graph', fontweight='bold')
ax.set_xticks(x_labels)
ax.set_xticklabels(models.keys())
ax.legend()
plt.ylim(0, 1.08)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

#  Confusion Matrices Graph
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (name, cm) in zip(axes, cm_dict.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()

#  Combined ROC Curves
plt.figure(figsize=(9, 7))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
plt.title(' ROC Curve All Compare', fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

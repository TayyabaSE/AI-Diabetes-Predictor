# AI-Diabetes-Predictor
AI based project for predicting diabetes using machine learning models.
import pandas as pd

# Load dataset
df = pd.read_csv("/content/diabetes.csv")
# Print first 10 rows
print("First 10 rows of dataset:")
print(df.head(10))
# Dataset info
print("\nDataset Info:")
print(df.info())
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())
import numpy as np
from sklearn.impute import SimpleImputer

# Replace 0 with NaN in medical columns
cols_with_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Impute missing values with mean
imputer = SimpleImputer(strategy="mean")
df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

# Check missing values
print("\nMissing values after imputation:")
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplots for numeric features
for col in cols_with_zero + ["Age","Pregnancies","DiabetesPedigreeFunction"]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Outlier treatment using IQR method
for col in cols_with_zero + ["Age","Pregnancies","DiabetesPedigreeFunction"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop("Outcome", axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

print("\nScaled dataset (first 5 rows):")
print(df_scaled.head())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Histogram of Glucose
sns.histplot(df["Glucose"], kde=True)
plt.title("Glucose Distribution")
plt.show()

# Pairplot for selected features
sns.pairplot(df[["Glucose","BMI","Age","Outcome"]], hue="Outcome")
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow Method
inertia = []
K = range(2,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Fit KMeans with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df["Cluster"] = clusters

# Evaluate
sil_score = silhouette_score(df_scaled, clusters)
print("\nSilhouette Score:", sil_score)
print("\nCluster distribution:")
print(df["Cluster"].value_counts())

# Scatter plot clusters
sns.scatterplot(x=df_scaled["Glucose"], y=df_scaled["BMI"], hue=df["Cluster"], palette="Set1")
plt.title("Clusters based on Glucose & BMI")
plt.show()

# Compare clusters with Outcome
ct = pd.crosstab(df["Cluster"], df["Outcome"])
print("\nCluster vs Outcome comparison:")
print(ct)

fig, axes = plt.subplots(1,2, figsize=(12,5))

# Before clustering
sns.scatterplot(x=df_scaled["Glucose"], y=df_scaled["BMI"], ax=axes[0])
axes[0].set_title("BEFORE Clustering")

# After clustering
sns.scatterplot(x=df_scaled["Glucose"], y=df_scaled["BMI"], hue=df["Cluster"], palette="Set1", ax=axes[1])
axes[1].set_title("AFTER Clustering")

plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2)
hc_clusters = hc.fit_predict(df_scaled)
df["HC_Cluster"] = hc_clusters

print("\nHierarchical Clustering vs Outcome:")
print(pd.crosstab(df["HC_Cluster"], df["Outcome"]))


from google.colab import output
output.enable_custom_widget_manager()

import ipywidgets as widgets
from IPython.display import display, HTML

# Header design
display(HTML("""
<div style='background:linear-gradient(90deg,#007acc,#00c6ff);padding:20px;border-radius:12px;text-align:center;'>
  <h1 style='color:white;font-family:Arial;'>💙 Diabetes Risk Clustering UI</h1>
  <p style='color:#f0f0f0;font-size:16px;'>Enter patient data using sliders below</p>
</div>
"""))

# Sliders (larger size)
preg = widgets.IntSlider(description='Pregnancies', min=0, max=15, value=3, layout=widgets.Layout(width='400px'))
glucose = widgets.IntSlider(description='Glucose Level', min=0, max=200, value=150, layout=widgets.Layout(width='400px'))
bp = widgets.IntSlider(description='Blood Pressure', min=50, max=150, value=80, layout=widgets.Layout(width='400px'))
skin = widgets.IntSlider(description='Skin Thickness', min=0, max=50, value=25, layout=widgets.Layout(width='400px'))
insulin = widgets.IntSlider(description='Insulin Level', min=0, max=300, value=100, layout=widgets.Layout(width='400px'))
bmi = widgets.FloatSlider(description='BMI', min=15, max=50, value=28.5, layout=widgets.Layout(width='400px'))
dpf = widgets.FloatSlider(description='Diabetes Pedigree', min=0.0, max=2.0, step=0.01, value=0.45, layout=widgets.Layout(width='400px'))
age = widgets.IntSlider(description='Age', min=18, max=80, value=35, layout=widgets.Layout(width='400px'))

# Predict button (big, rounded)
predict_btn = widgets.Button(
    description='🔍 Predict Cluster',
    button_style='info',
    layout=widgets.Layout(width='250px', height='50px')
)

# Output box
output_box = widgets.Output()

# Prediction function
def predict_cluster(b):
    with output_box:
        output_box.clear_output()
        cluster = 1  # Replace with your trained model prediction
        display(HTML(f"""
        <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center;margin-top:15px;'>
          <h2 style='color:#008000;font-family:Arial;'>✅ Patient belongs to Cluster: {cluster}</h2>
        </div>
        """))

predict_btn.on_click(predict_cluster)

# Layout (vertical stack, bigger spacing)
ui = widgets.VBox([
    preg, glucose, bp, skin, insulin, bmi, dpf, age,
    widgets.VBox([predict_btn, output_box])
])
display(ui)

from google.colab import output
output.enable_custom_widget_manager()

import ipywidgets as widgets
from IPython.display import display, HTML

# Background screen color
display(HTML("""
<style>
body {
    background-color: #f9f9ff; /* light lavender background */
}
.custom-btn {
    background: linear-gradient(90deg,#007acc,#00c6ff);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 30px;
    font-size: 16px;
    cursor: pointer;
    transition: 0.3s;
}
.custom-btn:hover {
    background: linear-gradient(90deg,#00c6ff,#007acc);
    box-shadow: 0px 0px 12px rgba(0, 118, 255, 0.7);
}
</style>
<div style='text-align:center;padding:20px;'>
  <h1 style='color:#007acc;font-family:Arial;'>💙 Diabetes Risk Clustering UI</h1>
  <p style='color:#555;font-size:16px;'>Enter patient data using sliders below</p>
</div>
"""))

# Sliders
preg = widgets.IntSlider(description='Pregnancies', min=0, max=15, value=3, layout=widgets.Layout(width='500px'))
glucose = widgets.IntSlider(description='Glucose Level', min=0, max=200, value=150, layout=widgets.Layout(width='500px'))
bp = widgets.IntSlider(description='Blood Pressure', min=50, max=150, value=80, layout=widgets.Layout(width='500px'))
skin = widgets.IntSlider(description='Skin Thickness', min=0, max=50, value=25, layout=widgets.Layout(width='500px'))
insulin = widgets.IntSlider(description='Insulin Level', min=0, max=300, value=100, layout=widgets.Layout(width='500px'))
bmi = widgets.FloatSlider(description='BMI', min=15, max=50, value=28.5, layout=widgets.Layout(width='500px'))
dpf = widgets.FloatSlider(description='Diabetes Pedigree', min=0.0, max=2.0, step=0.01, value=0.45, layout=widgets.Layout(width='500px'))
age = widgets.IntSlider(description='Age', min=18, max=80, value=35, layout=widgets.Layout(width='500px'))

# Predict button (centered with hover effect)
predict_btn = widgets.Button(description='Predict Cluster', layout=widgets.Layout(width='250px', height='50px'))
predict_btn.add_class("custom-btn")

output_box = widgets.Output()

def predict_cluster(b):
    with output_box:
        output_box.clear_output()
        cluster = 1  # Replace with your trained model prediction
        display(HTML(f"""
        <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center;margin-top:15px;'>
          <h2 style='color:#008000;font-family:Arial;'>✅ Patient belongs to Cluster: {cluster}</h2>
        </div>
        """))

predict_btn.on_click(predict_cluster)

# Layout
ui = widgets.VBox([
    preg, glucose, bp, skin, insulin, bmi, dpf, age,
    widgets.VBox([predict_btn, output_box], layout=widgets.Layout(align_items='center'))
])
display(ui)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split dataset
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model (predicts majority class)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("Baseline Accuracy (Before Training):", accuracy_score(y_test, y_pred_dummy))

from sklearn.linear_model import LogisticRegression

# Train a real model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

print("Trained Model Accuracy (After Training):", accuracy_score(y_test, y_pred_model))

print("\nCluster vs Outcome (KMeans):")
print(pd.crosstab(df["Cluster"], df["Outcome"]))

print("\nCluster vs Outcome (Hierarchical):")
print(pd.crosstab(df["HC_Cluster"], df["Outcome"]))

# Visual comparison
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1,2, figsize=(12,5))

sns.scatterplot(x=df_scaled["Glucose"], y=df_scaled["BMI"], hue=df["Cluster"], palette="Set1", ax=axes[0])
axes[0].set_title("KMeans Clustering")

sns.scatterplot(x=df_scaled["Glucose"], y=df_scaled["BMI"], hue=df["HC_Cluster"], palette="Set2", ax=axes[1])
axes[1].set_title("Hierarchical Clustering")

plt.show()



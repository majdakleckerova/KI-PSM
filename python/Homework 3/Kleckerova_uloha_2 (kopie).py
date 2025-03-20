## Načtení knihoven, modulů
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## Styl grafů
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

## 2.1. Načtení, průzkum dat
winequality = pd.read_csv("winequality-red.csv", delimiter=";")
vstupni_promenne = winequality.columns[:-1].to_list()
cilova_promenna = winequality.columns[-1]
print(f"\nProzkoumání datasetu winequality-red.csv:")
print(winequality.describe())

## Vizualizace
### Počet případů v datasetu
def plot_class_distribution(nazev_datasetu: str, data, cilova_promenna: str):
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=data[cilova_promenna], hue= data[cilova_promenna], palette=["hotpink","deeppink","mediumvioletred","orchid","magenta","darkmagenta"], legend=False)
    plt.title(f"Rozložení případů v datasetu ({nazev_datasetu})")
    plt.xlabel("Cílová proměnná")
    plt.ylabel("Počet vzorků")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    unique_classes = sorted(data[cilova_promenna].unique())
    ax.set_xticks(range(len(unique_classes)))
    total = len(data)
    for p in ax.patches:
        relativni_cetnost = f"{100 * p.get_height()/total:.1f} %"
        ax.annotate(relativni_cetnost, (p.get_x() + p.get_width()/2, p.get_height() + 3),
                    ha = "center", va = "bottom", fontweight = "bold")
    plt.savefig(f"graf_{nazev_datasetu}_1.png", dpi=300, bbox_inches='tight')

### Korelační matice mezi atributy
def plot_correlation_matrix(nazev_datasetu: str, data):
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.corr(), annot=True, cmap="PuRd", fmt=".2f", linewidths=0.5)
    plt.title(f"Korelační matice atributů ({nazev_datasetu})")
    plt.savefig(f"graf_{nazev_datasetu}_2.png", dpi=300, bbox_inches='tight')

plot_class_distribution("winequality-red.csv", winequality, "quality")
plot_correlation_matrix("winequality-red.csv", winequality)


## 2.2. Předzpracování dat
def prepare_data(df, target_col, test_size=0.3, random_state=42):
    df = df.copy()  
    ## Zpracování chybějících hodnot 
    positive_cols = [col for col in df.columns if df[col].min() > 0 and col != target_col]
    for col in positive_cols:
        df[col] = df[col].replace(0, np.nan)  
        df[col] = df[col].fillna(df[col].median())  
    # Alternativně: doplnění všech sloupců průměrem
    df.fillna(df.mean(), inplace=True)
    ## Rozdělení na vstupní proměnné (X) a cílovou proměnnou (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    ## Normalizace numerických atributů
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ## Rozdělení na trénovací a testovací sadu
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

## winequality-red.csv
X_train, X_test, y_train, y_test = prepare_data(winequality, "quality")

## 2.3, 2.4. Vytvoření modelů, predikcí na testovacích datech
def create_models(X_train, X_test, y_train, y_test):
    ## Rozhodovací strom
    model_rs = DecisionTreeClassifier(random_state=42)
    model_rs.fit(X_train, y_train)
    y_pred_rs = model_rs.predict(X_test)
    y_prob_rs = model_rs.predict_proba(X_test)[:, 1]
    ## Random forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    return model_rs, model_rf, y_pred_rs, y_pred_rf, y_prob_rs, y_prob_rf

model_rs, model_rf, y_pred_rs, y_pred_rf, y_prob_rs, y_prob_rf = create_models(X_train, X_test, y_train, y_test)

## 2.5. Vyhodnocení pomocí metrik (MSE, RMSE, R^2)
def vypocitej_metriky(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

mse_rs, rmse_rs, r2_rs = vypocitej_metriky(y_test, y_pred_rs, "Rozhodovací strom")
mse_rf, rmse_rf, r2_rf = vypocitej_metriky(y_test, y_pred_rf, "Random Forest")

def porovnej_modely(vysledky):
    nejlepsi_model = None
    nejlepsi_score = np.inf  
    for model, (mse, rmse, r2) in vysledky.items():
        print(f"{model}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
        score = mse + rmse - r2  # Chceme minimalizovat MSE & RMSE, ale maximalizovat R²
        if score < nejlepsi_score:
            nejlepsi_score = score
            nejlepsi_model = model
    print(f"\nLepší model: {nejlepsi_model}\n")
    return nejlepsi_model

metrics_df = {
    "Rozhodovací strom": (mse_rs, rmse_rs, r2_rs),
    "Random Forest": (mse_rf, rmse_rf, r2_rf)
}
nejlepsi_model = porovnej_modely(metrics_df)

## 2.6. Vizualizace metrik
def vykresli_porovnani_modelu(nazev_datasetu:str, mse_rs, rmse_rs, r2_rs, mse_rf, rmse_rf, r2_rf):
    metriky = ["MSE", "RMSE", "R²"]
    hodnoty_rs = [mse_rs, rmse_rs, r2_rs]
    hodnoty_rf = [mse_rf, rmse_rf, r2_rf]
    x = np.arange(len(metriky))  
    width = 0.3  
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Porovnání výkonnosti modelů", fontsize=16, fontweight="bold")
    for i, ax in enumerate(axes):
        ax.bar(x[i] - width/2, hodnoty_rs[i], width, label="Rozhodovací strom", color="hotpink")
        ax.bar(x[i] + width/2, hodnoty_rf[i], width, label="Random Forest", color="mediumvioletred")
        ax.set_title(metriky[i], fontweight="bold")
        ax.set_xticks([])  
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.text(x[i] - width/2, hodnoty_rs[i] + 0.01, f"{hodnoty_rs[i]:.3f}", ha="center", fontweight="bold")
        ax.text(x[i] + width/2, hodnoty_rf[i] + 0.01, f"{hodnoty_rf[i]:.3f}", ha="center", fontweight="bold")
    axes[0].set_ylabel("Hodnota metriky")
    axes[0].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(f"graf_{nazev_datasetu}_3.png", dpi=300, bbox_inches='tight')

vykresli_porovnani_modelu("winequality-red.csv", mse_rs, rmse_rs, r2_rs, mse_rf, rmse_rf, r2_rf)
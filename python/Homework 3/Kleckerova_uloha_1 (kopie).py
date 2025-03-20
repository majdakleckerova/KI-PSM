#1.1. Načtení dat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

## Načtení dat
diabetes = pd.read_csv("diabetes.csv")
framingham = pd.read_csv("framingham.csv")

## Nastavení stylu grafů
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print(f"\n Prozkoumání datasetu diabetes.csv:")
print(diabetes.describe())
print(f"\n Prozkoumání datasetu framingham.csv:")
print(framingham.describe())

### Vizualizace pro představu o datasetu
## Počet případů v datasetu
def plot_class_distribution(nazev_datasetu: str, data, cilova_promenna: str):
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=data[cilova_promenna], hue=data[cilova_promenna], palette=["darkviolet", "hotpink"], legend=False)
    plt.title(f"Rozložení případů v datasetu ({nazev_datasetu})")
    plt.xticks(ticks=[0, 1], labels=["Nevyskytuje se", "Vyskytuje se"])
    plt.xlabel("Cílová proměnná")
    plt.ylabel("Počet vzorků")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    total = len(data)
    for p in ax.patches:
        relativni_cetnost = f"{100 * p.get_height()/total:.1f} %"
        ax.annotate(relativni_cetnost, (p.get_x() + p.get_width()/2, p.get_height() + 3),
                    ha = "center", va = "bottom", fontweight = "bold")
    plt.savefig(f"graf_{nazev_datasetu}_1.png", dpi=300, bbox_inches='tight')

## Korelační matice mezi atributy
def plot_correlation_matrix(nazev_datasetu: str, data):
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.corr(), annot=True, cmap="PuRd", fmt=".2f", linewidths=0.5)
    plt.title(f"Korelační matice atributů ({nazev_datasetu})")
    plt.savefig(f"graf_{nazev_datasetu}_2.png", dpi=300, bbox_inches='tight')

## Distribuce atributů mezi třídami
def plot_feature_distributions(nazev_datasetu: str, data, features, target=None):
    if target is None:
        target = data.iloc[:, -1]
    num_features = len(features)
    rows = (num_features // 3) + (num_features % 3 > 0)  
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))  
    fig.suptitle(f"Distribuce atributů mezi třídami ({nazev_datasetu})", fontsize=14, y = 1.02)
    axes = axes.flatten() 
    for i, feature in enumerate(features):
        sns.histplot(data, x=feature, hue=target, kde=True, palette=["darkviolet", "hotpink"], ax=axes[i])
        axes[i].set_title(f"Distribuce: {feature}")
        axes[i].grid(axis="x", linestyle="--", alpha=0.7)
    for j in range(i + 1, len(axes)):  
        fig.delaxes(axes[j])  
    plt.tight_layout()
    plt.savefig(f"graf_{nazev_datasetu}_3.png", dpi=300, bbox_inches='tight')


## diabetes.csv
plot_class_distribution("diabetes.csv", diabetes, "Outcome")
plot_correlation_matrix("diabetes.csv", diabetes)
vstupni_promenne = diabetes.columns[:-1].tolist()
plot_feature_distributions("diabetes.csv", diabetes, vstupni_promenne)

## framingham.csv
plot_class_distribution("framingham.csv", framingham, "TenYearCHD")
plot_correlation_matrix("framingham.csv", framingham)
vstupni_promenne = framingham.columns[:-1].tolist()
plot_feature_distributions("framingham.csv", framingham, vstupni_promenne)

# 1.2. Předzpracování dat
def prepare_data(df, target_col, test_size=0.3, random_state=42):
    df = df.copy()  
    ## Zpracování chybějících hodnot 
    positive_cols = [col for col in df.columns if df[col].min() > 0 and col != target_col]
    for col in positive_cols:
        df[col] = df[col].replace(0, np.nan)  
        df[col] = df[col].fillna(df[col].median()) 
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

## diabetes.csv
X_train, X_test, y_train, y_test = prepare_data(diabetes, "Outcome")

## framingham.csv
X_train_2, X_test_2, y_train_2, y_test_2 = prepare_data(framingham, "TenYearCHD")

# 1.3. , 1.4. Vytvoření modelů, testování
def create_models(X_train, X_test, y_train, y_test):
    ## Logistický model
    model_1 = LogisticRegression(max_iter = 1000, random_state=42)
    model_1.fit(X_train, y_train)
    y_pred_1 = model_1.predict(X_test)
    y_prob_1 = model_1.predict_proba(X_test)[:, 1]
    #print(classification_report(y_test, y_pred_1))

    ## Rozhodovací strom
    model_2 = DecisionTreeClassifier(random_state=42)
    model_2.fit(X_train, y_train)
    y_pred_2 = model_2.predict(X_test)
    y_prob_2 = model_2.predict_proba(X_test)[:, 1]
    #print(classification_report(y_test, y_pred_2))

    ## Random forest
    model_3 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_3.fit(X_train, y_train)
    y_pred_3 = model_3.predict(X_test)
    y_prob_3 = model_3.predict_proba(X_test)[:, 1]
    #print(classification_report(y_test, y_pred_3))

    return model_1, model_2, model_3, y_pred_1, y_pred_2, y_pred_3, y_prob_1, y_prob_2, y_prob_3

model_1_1, model_2_1, model_3_1, y_pred_1_1, y_pred_2_1, y_pred_3_1, y_prob_1_1, y_prob_2_1, y_prob_3_1 = create_models(X_train, X_test, y_train, y_test)
model_1_2, model_2_2, model_3_2, y_pred_1_2, y_pred_2_2, y_pred_3_2, y_prob_1_2, y_prob_2_2, y_prob_3_2 = create_models(X_train_2, X_test_2, y_train_2, y_test_2)

# 1.5. Vyhodnocení, srovnání modelů
## Uložení dat do slovníků
y_pred_1 = {
    "Logistická regrese": y_pred_1_1,
    "Rozhodovací strom": y_pred_2_1,
    "Random Forest": y_pred_3_1}
y_prob_1 = {
    "Logistická regrese": y_prob_1_1,
    "Rozhodovací strom": y_prob_2_1,
    "Random Forest": y_prob_3_1}

y_pred_2 = {
    "Logistická regrese": y_pred_1_2,
    "Rozhodovací strom": y_pred_2_2,
    "Random Forest": y_pred_3_2}
y_prob_2 = {
    "Logistická regrese": y_prob_1_2,
    "Rozhodovací strom": y_prob_2_2,
    "Random Forest": y_prob_3_2}


## Výpočet metrik
def vypocti_metriky(nazev_datasetu:str, y_test, y_pred, y_prob):
    metrics = []
    for model_name, y_pred_values in y_pred.items():
        acc = accuracy_score(y_test, y_pred_values)
        prec = precision_score(y_test, y_pred_values)
        rec = recall_score(y_test, y_pred_values)
        f1 = f1_score(y_test, y_pred_values)
        fpr, tpr, _ = roc_curve(y_test, y_prob[model_name])
        auc_score = auc(fpr, tpr)
        metrics.append([model_name, acc, prec, rec, f1, auc_score]) 
    metrics_df = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall (citlivost)", "F1-score", "AUC"])
    print(f"\n({nazev_datasetu}): Srovnání výkonu modelů:")
    print(tabulate(metrics_df, headers="keys", tablefmt="pretty"))
    return metrics_df


## Srovnání modelů podle hodnot metrik
def evaluate_models(nazev_datasetu:str, metrics_df, w_acc=0.2, w_prec=0.2, w_rec=0.2, w_f1=0.2, w_auc=0.2):
    ## 1. Nejvyšší průměr metrik
    metrics_df["Průměr"] = metrics_df.iloc[:, 1:].mean(axis=1)
    nejlepsi_model_1 = metrics_df.loc[metrics_df["Průměr"].idxmax(), "Model"]
    ## 2. Nejvyšší hodnota Accuracy
    nejlepsi_model_2 = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
    ## 3. Nejvyšší hodnota AUC
    nejlepsi_model_3 = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
    ## 4. Nejvyšší vážený průměr metrik
    metrics_df["Skóre"] = (
        metrics_df["Accuracy"] * w_acc + 
        metrics_df["Precision"] * w_prec + 
        metrics_df["Recall (citlivost)"] * w_rec + 
        metrics_df["F1-score"] * w_f1 + 
        metrics_df["AUC"] * w_auc)
    nejlepsi_model_4 = metrics_df.loc[metrics_df["Skóre"].idxmax(), "Model"]
    
    print(f"\n({nazev_datasetu}): Nejlepší model podle:")
    print(f"\n1. Průměrné hodnoty metrik: {nejlepsi_model_1}")
    print(f"\n2. Přesnosti (metrika Accuracy): {nejlepsi_model_2}")
    print(f"\n3. Hodnoty AUC: {nejlepsi_model_3}")
    print(f"\n4. Váženého průměru metrik: {nejlepsi_model_4}")
    return nejlepsi_model_1, nejlepsi_model_2, nejlepsi_model_3, nejlepsi_model_4


## Vizualizace modelů
def visualize_models(nazev_datasetu:str, y_test, y_pred, y_prob, metrics_df, model_l, model_rs, model_rf, vstupni_promenne:list):
    print(f"\n({nazev_datasetu}): Vizualizace:\n")

    ## Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(20,5))  
    fig.suptitle(f"Confusion Matrix ({nazev_datasetu})")
    for i, (model_name, y_pred_values) in enumerate(y_pred.items()):
        cm = confusion_matrix(y_test, y_pred_values)
        sns.heatmap(cm, annot=True, fmt='d', cmap="RdPu", xticklabels=['No Diabetes', 'Diabetes'], 
                    yticklabels=['No Diabetes', 'Diabetes'], ax=axes[i])
        axes[i].set_xlabel('Predikované')
        axes[i].set_ylabel('Skutečné')
        axes[i].set_title(f"{model_name}") 
    plt.tight_layout()
    plt.savefig(f"graf_{nazev_datasetu}_4.png", dpi=300, bbox_inches='tight')

    ## ROC křivka 
    plt.figure(figsize=(8,6))
    colors = ["hotpink","paleturquoise", "mediumaquamarine"]
    for (model_name, y_prob_values), color in zip(y_prob.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob_values)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,color = color, label=f"{model_name} (AUC = {roc_auc:.2f})", linewidth = 2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth = 1)  
    plt.title(f"ROC křivka ({nazev_datasetu})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"graf_{nazev_datasetu}_5.png", dpi=300, bbox_inches='tight')

    ## Vizualizace metrik 
    metriky = ["Accuracy", "Precision", "Recall (citlivost)", "F1-score", "AUC"]
    barvy = ["hotpink", "paleturquoise", "mediumaquamarine"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  
    fig.suptitle(f"Sloupcový graf metrik ({nazev_datasetu})")
    for i, metrika in enumerate(metriky):
        ax = axes[i]
        bars = ax.bar(metrics_df["Model"], metrics_df[metrika], color=barvy)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", 
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metrika)
        ax.set_ylim(0, 1) 
        ax.set_xticks(range(len(models))) 
        ax.set_xticklabels(metrics_df["Model"], rotation=20)
    plt.tight_layout()
    plt.savefig(f"graf_{nazev_datasetu}_6.png", dpi=300, bbox_inches='tight')

    ## Významnost atributů
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))  
    fig.suptitle(f"Významnost atributů ({nazev_datasetu})")
    num_features = len(vstupni_promenne)
    coef_values = model_l.coef_[0]  
    if len(coef_values) != num_features:
        print(f"POZOR: Model má {len(coef_values)} koeficientů, ale dataset má {num_features} vstupních proměnných!")
        coef_values = coef_values[:num_features]  # Ořízneme na shodný počet
    # Logistická regrese (koeficienty)
    axes[0].barh(vstupni_promenne[:len(coef_values)], coef_values, color="hotpink")
    axes[0].set_title("Logistická regrese")
    axes[0].set_xlabel("Koeficient")
    # Rozhodovací strom (Feature Importance)
    importances_tree = model_rs.feature_importances_
    if len(importances_tree) != num_features:
        print(f"POZOR: Rozhodovací strom má {len(importances_tree)} atributů, ale dataset má {num_features} vstupních proměnných!")
        importances_tree = importances_tree[:num_features]
    axes[1].barh(vstupni_promenne[:len(importances_tree)], importances_tree, color="paleturquoise")
    axes[1].set_title("Rozhodovací strom")
    axes[1].set_xlabel("Význam atributu")
    # Random Forest (Feature Importance)
    importances_rf = model_rf.feature_importances_
    if len(importances_rf) != num_features:
        print(f"POZOR: Random Forest má {len(importances_rf)} atributů, ale dataset má {num_features} vstupních proměnných!")
        importances_rf = importances_rf[:num_features]
    axes[2].barh(vstupni_promenne[:len(importances_rf)], importances_rf, color="mediumaquamarine")
    axes[2].set_title("Random Forest")
    axes[2].set_xlabel("Význam atributu")
    plt.tight_layout()
    plt.savefig(f"graf_{nazev_datasetu}_7.png", dpi=300, bbox_inches='tight')

# diabetes.csv 
metrics_df = vypocti_metriky("diabetes.csv", y_test, y_pred_1, y_prob_1)
models = metrics_df["Model"]
evaluate_models("diabetes.csv", metrics_df)
feature_names = diabetes.columns[:-1].tolist()
visualize_models("diabetes.csv", y_test, y_pred_1, y_prob_1, metrics_df, model_1_1, model_2_1, model_3_1, feature_names)

# framingham.csv
metrics_df_2 = vypocti_metriky("framingham.csv", y_test_2, y_pred_2, y_prob_2)
models_2 = metrics_df_2["Model"]
evaluate_models("framingham.csv", metrics_df_2)
feature_names_2 = framingham.columns[:-1].tolist()
visualize_models("framingham.csv", y_test_2, y_pred_2, y_prob_2, metrics_df_2, model_1_2, model_2_2, model_3_2, feature_names_2)


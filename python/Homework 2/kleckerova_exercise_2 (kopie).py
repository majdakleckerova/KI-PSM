# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro, skew

# Nastavení stylu grafů
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
barvy = ["deeppink", "fuchsia", "magenta", "hotpink", "violet"]

# 1. Načtení datové sady
House_price = pd.read_csv("house_data.csv", delimiter=",")
print(House_price.head())
print(House_price.info())
print(House_price.describe())

# 2. Průzkum dat
# -----------------------------------------------------------------------------------
print("\n2. Průzkum dat")

# snaha o predikci ceny domu (MEDV)
# Popis proměnných:
# CRIM: kriminalita v oblasti
# ZN: procento pozemků pro velké domy
# INDUS: procento pro obchodní plochy
# CHAS: blízkost k řece (binární, 0/1)
# NX: znečištění ovzduší
# RM: průměrný počet místností v domě
# AGE: procento starších domů
# DIS: vzdálenost do zaměstnání
# RAD: index dostupnosti dálnice
# TAX: výše daně z nemovitosti
# PTRATIO: poměr studentů na učitele
# B: etnický ukazatel (čím vyšší, tím více bělochů)
# LSTAT: procento lidí s nízkým sociálním statusem
# MEDV: cena domu (cílová proměnná)

cilova_promenna = "MEDV"
vstupni_promenne = ["CRIM", "ZN", "INDUS", "CHAS", "NX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Vizualizace vztahů mezi proměnnými a cílovou proměnnou
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle("Scatter ploty vstupních proměnných vs. Cena domu (MEDV)")

for i, feature in enumerate(vstupni_promenne):
    row, col = divmod(i, 4)
    axes[row, col].scatter(House_price[feature], House_price[cilova_promenna], color='orchid', alpha=0.75)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel(cilova_promenna)
    axes[row, col].set_title(f"{cilova_promenna} = f({feature})")
    axes[row, col].set_facecolor('#E6E6FA')

# Skrytí prázdných os
for j in range(len(vstupni_promenne), 16):
    row, col = divmod(j, 4)
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('graf_2_1.png')

# Korelační matice
plt.figure(figsize=(12, 8))
sns.heatmap(House_price.corr(), annot=True, cmap='RdPu', fmt=".2f", linewidths=0.5)
plt.title("Korelační matice", fontweight='bold')
plt.tight_layout()
plt.savefig('graf_2_2.png')

# VIF (Variance Inflation Factor) - kontrola multikolinearity
X = add_constant(House_price.drop(columns=[cilova_promenna]))
vif_data = pd.DataFrame()
vif_data["Vstupní proměnná"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF hodnoty pro vstupní proměnné:")
print(vif_data.sort_values('VIF', ascending=False))

# Na základě grafů, korelací a VIF odstranění neinformativních sloupců
drop_columns = ["CHAS", "B", "RAD", "INDUS", "ZN"]
House_price_2 = House_price.drop(columns=drop_columns)
print("\nRozměry původních dat:", House_price.shape)
print("Rozměry po odstranění neinformativních sloupců:", House_price_2.shape)
print("\nData po odstranění neinformativních sloupců:")
print(House_price_2.head())

# 3. Analýza dat
# -----------------------------------------------------------------------------------
print("\n3. Analýza dat")

# 3.1 Kontrola multikolinearity po odstranění sloupců
X_2 = add_constant(House_price_2.drop(columns=[cilova_promenna]))
vif_data_2 = pd.DataFrame()
vif_data_2["Vstupní proměnná"] = X_2.columns
vif_data_2["VIF"] = [variance_inflation_factor(X_2.values, i) for i in range(X_2.shape[1])]
print("\nVIF hodnoty po odstranění neinformativních sloupců:")
print(vif_data_2.sort_values('VIF', ascending=False))

# 3.2 Analýza nutnosti normalizace
print("\nStatistiky před normalizací:")
print(House_price_2.describe())

# Boxplot pro vizualizaci rozsahu hodnot
plt.figure(figsize=(10, 5))
sns.boxplot(data=House_price_2.drop(columns=[cilova_promenna]), color="orchid")
plt.xticks(rotation=45)
plt.title("Rozložení hodnot jednotlivých proměnných", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig('graf_2_3.png')

# Normalizace dat pomocí MinMaxScaler
scaler = MinMaxScaler()
House_price_scaled = House_price_2.copy()
features_to_scale = House_price_scaled.drop(columns=[cilova_promenna]).columns
House_price_scaled[features_to_scale] = scaler.fit_transform(House_price_scaled[features_to_scale])
print("\nStatistiky po normalizaci:")
print(House_price_scaled.describe())

# 3.3 Analýza potřeby transformace dat
# Test normality
print("\nTesty normality (Shapiro-Wilk):")
for sloupec in House_price_2.drop(columns=[cilova_promenna]).columns:
    stat, p = shapiro(House_price_2[sloupec])
    print(f"{sloupec}: p-value = {p:.8f}")

# Šikmost
print("\nŠikmost:")
for sloupec in House_price_2.drop(columns=[cilova_promenna]).columns:
    print(f"{sloupec}: Šikmost = {skew(House_price_2[sloupec]):.2f}")

# Histogramy distribucí před transformací
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Distribuce vstupních proměnných před transformací", fontsize=16, fontweight="bold")
columns = House_price_2.drop(columns=[cilova_promenna]).columns
for i, col in enumerate(columns):
    row, col_idx = divmod(i, 4)
    sns.histplot(House_price_2[col], ax=axes[row, col_idx], kde=True, color="deeppink")
    axes[row, col_idx].set_title(col, fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('graf_2_4.png')

# Transformace šikmých proměnných
# Na základě testů aplikujeme transformace
dataset = House_price_scaled.copy()

# Power transformace CRIM (odstranění šikmosti)
pt = PowerTransformer(method='yeo-johnson')
dataset['CRIM_trans'] = pt.fit_transform(dataset[['CRIM']])

# Log transformace LSTAT (negativní šikmost)
dataset['LSTAT_log'] = np.log1p(dataset['LSTAT'])

# Standardizace AGE
scaler_std = StandardScaler()
dataset['AGE_std'] = scaler_std.fit_transform(dataset[['AGE']])

# Transformace TAX
dataset['TAX_log'] = np.log1p(dataset['TAX'])

# Odstranění původních netransformovaných sloupců
dataset_transformed = dataset.drop(columns=['CRIM', 'LSTAT', 'AGE', 'TAX'])

# Histogramy po transformaci
transformed_cols = ['CRIM_trans', 'LSTAT_log', 'AGE_std', 'TAX_log']
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Distribuce po transformaci", fontsize=16, fontweight="bold")
for i, col in enumerate(transformed_cols):
    row, col_idx = divmod(i, 2)
    sns.histplot(dataset_transformed[col], ax=axes[row, col_idx], kde=True, color="fuchsia")
    axes[row, col_idx].set_title(col, fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('graf_2_5.png')

# 4. Vytvoření regresních modelů
# -----------------------------------------------------------------------------------
print("\n4. Vytvoření regresních modelů")
print("-----------------------------------------------------------------------------------")

# Rozdělení dat na trénovací a testovací množinu (70/30)
X = dataset_transformed.drop(columns=[cilova_promenna])
y = dataset_transformed[cilova_promenna]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Trénovací data: {X_train.shape}, Testovací data: {X_test.shape}")

# Definice různých kombinací proměnných pro modely
modely = {
    "Model 1 (Full)": list(X.columns),
    "Model 2 (Bez NX)": [col for col in X.columns if col != 'NX'],
    "Model 3 (Nejdůležitější proměnné)": ['RM', 'LSTAT_log', 'PTRATIO', 'DIS']
}

# Vytvoření a vyhodnocení modelů
vysledky_modelu = []
modely_objekty = {}
predikce = {}

for nazev_modelu, promenne in modely.items():
    print(f"\nVytváření modelu: {nazev_modelu}")
    print("Použité proměnné:", promenne)
    
    # Výběr proměnných pro tento model
    X_train_model = X_train[promenne]
    X_test_model = X_test[promenne]
    
    # Přidání konstanty pro statsmodels
    X_train_const = sm.add_constant(X_train_model)
    X_test_const = sm.add_constant(X_test_model)
    
    # Vytvoření OLS modelu
    ols_model = sm.OLS(y_train, X_train_const).fit()
    
    # Predikce
    y_pred = ols_model.predict(X_test_const)
    predikce[nazev_modelu] = y_pred
    
    # Výpočet metrik
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Uložení výsledků
    vysledky_modelu.append({
        "Model": nazev_modelu,
        "RMSE": rmse,
        "R²": r2,
        "Počet proměnných": len(promenne)
    })
    
    # Uložení modelu
    modely_objekty[nazev_modelu] = ols_model
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("Statistické shrnutí modelu:")
    print(ols_model.summary().tables[1])

# Vytvoření přehledné tabulky výsledků
df_vysledky = pd.DataFrame(vysledky_modelu)
print("\nPorovnání modelů:")
print(df_vysledky)

# Vizualizace metrik modelů
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_vysledky = pd.DataFrame(vysledky_modelu)
print("\nPorovnání modelů:")
print(df_vysledky)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(vysledky_modelu))
modely = [v["Model"] for v in vysledky_modelu]
rmse_bars = axes[0].bar(x, [v["RMSE"] for v in vysledky_modelu], color='deeppink')
axes[0].set_title('RMSE modelů', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(modely, rotation=45, ha='right')
axes[0].bar_label(rmse_bars, fmt='%.3f', padding=3)
r2_bars = axes[1].bar(x, [v["R²"] for v in vysledky_modelu], color='fuchsia')
axes[1].set_title('R² modelů', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(modely, rotation=45, ha='right')
axes[1].bar_label(r2_bars, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig('graf_2_6.png')


# Analýza reziduí pro všechny modely
fig, axes = plt.subplots(len(modely), 2, figsize=(16, 6*len(modely)))
fig.suptitle('Analýza reziduí modelů', fontsize=16, fontweight='bold')
for i, (nazev_modelu, y_pred) in enumerate(predikce.items()):
    residuals = y_test - y_pred
    
    # Scatter plot reziduí
    axes[i, 0].scatter(y_pred, residuals, color='orchid', alpha=0.7)
    axes[i, 0].axhline(y=0, color='k', linestyle='-')
    axes[i, 0].set_title(f'{nazev_modelu}: Rezidua vs. Predikované hodnoty', fontweight='bold')
    axes[i, 0].set_xlabel('Predikované hodnoty', fontweight='bold')
    axes[i, 0].set_ylabel('Rezidua', fontweight='bold')
    
    # Histogram reziduí
    axes[i, 1].hist(residuals, bins=20, color='hotpink', alpha=0.7, density=True)
    sns.kdeplot(residuals, ax=axes[i, 1], color='deeppink')
    axes[i, 1].set_title(f'{nazev_modelu}: Distribuce reziduí', fontweight='bold')
    axes[i, 1].set_xlabel('Rezidua', fontweight='bold')
    axes[i, 1].set_ylabel('Hustota', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('graf_2_7.png')

barvy = ['orchid', 'deeppink', 'fuchsia']
# QQ ploty reziduí
fig, axes = plt.subplots(1, len(modely), figsize=(16, 5))
fig.suptitle('QQ ploty reziduí modelů', fontsize=16, fontweight='bold')
for i, (nazev_modelu, y_pred) in enumerate(predikce.items()):
    residuals = y_test - y_pred
    sm.qqplot(residuals, line='45', fit=True, ax=axes[i], color=barvy[i % len(barvy)])
    axes[i].set_title(f'{nazev_modelu}', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('graf_2_8.png')

# Scatter ploty skutečných vs. predikovaných hodnot
fig, axes = plt.subplots(1, len(modely), figsize=(18, 5))
fig.suptitle('Skutečné vs. predikované hodnoty', fontsize=16, fontweight='bold')
for i, (nazev_modelu, y_pred) in enumerate(predikce.items()):
    axes[i].scatter(y_test, y_pred, color=barvy[i], alpha=0.7)
    # Přidání diagonální čáry
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    axes[i].plot([min_val, max_val], [min_val, max_val], 'k--')
    axes[i].set_title(f'{nazev_modelu}: R² = {r2_score(y_test, y_pred):.4f}', fontweight='bold')
    axes[i].set_xlabel('Skutečné hodnoty', fontweight='bold')
    axes[i].set_ylabel('Predikované hodnoty', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('graf_2_9.png')

# Porovnání koeficientů modelů
# Získání koeficientů z každého modelu
koeficienty_modelu = {}
for nazev_modelu, model in modely_objekty.items():
    koef = model.params.drop('const') if 'const' in model.params.index else model.params
    koeficienty_modelu[nazev_modelu] = koef

# Příprava dat pro vizualizaci
all_features = set()
for koef in koeficienty_modelu.values():
    all_features.update(koef.index)

koef_df = pd.DataFrame(index=list(all_features))
for nazev_modelu, koef in koeficienty_modelu.items():
    for feature in all_features:
        if feature in koef.index:
            koef_df.loc[feature, nazev_modelu] = koef[feature]
        else:
            koef_df.loc[feature, nazev_modelu] = 0

# Vizualizace koeficientů všech modelů
plt.figure(figsize=(14, 8))
koef_df = koef_df.fillna(0)
koef_df = koef_df.reindex(koef_df.mean(axis=1).abs().sort_values(ascending=False).index)

ax = koef_df.plot(kind='bar', figsize=(14, 8), color=barvy[:len(modely)])
plt.title('Porovnání koeficientů modelů', fontweight='bold', fontsize=16)
plt.xlabel('Proměnná', fontweight='bold')
plt.ylabel('Hodnota koeficientu', fontweight='bold')
plt.legend(title='Model')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('graf_2_10.png')

# 5. Závěrečné vyhodnocení
# -----------------------------------------------------------------------------------
print("\n5. Závěrečné vyhodnocení")
print("-----------------------------------------------------------------------------------")

# Výběr nejlepšího modelu
best_model_index = df_vysledky['R²'].idxmax()
best_model = df_vysledky.iloc[best_model_index]

print(f"Nejlepší model dle R²: {best_model['Model']}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"R²: {best_model['R²']:.4f}")

# Interpretace výsledků
print("\nInterpretace výsledků:")
print(f"- Nejlepší model vysvětluje {best_model['R²']*100:.2f}% variability v datech.")
print(f"- Průměrná odchylka predikce je {best_model['RMSE']:.2f} jednotek.")

# Tabulka výsledků
vysledky_interpretace = []
for i, row in df_vysledky.iterrows():
    if row['Model'] == "Model 1 (Full)":
        interpretace = "Obsahuje všechny proměnné po transformaci. Vysoký R², ale riziko přeučení."
    elif row['Model'] == "Model 2 (Bez NX)":
        interpretace = "Odstraněna proměnná NX, která měla nízkou korelaci s cenou. Podobný výkon jako plný model."
    else:
        interpretace = "Pouze nejdůležitější prediktory, jednodušší model s dobrou vypovídací schopností."
    
    vysledky_interpretace.append([
        row['Model'], 
        f"{row['RMSE']:.4f}", 
        f"{row['R²']:.4f}", 
        interpretace
    ])

print("\nPorovnání modelů:")
print("-" * 90)
print(f"{'Model':<25} | {'RMSE':<10} | {'R²':<10} | {'Interpretace':<40}")
print("-" * 90)
for row in vysledky_interpretace:
    print(f"{row[0]:<25} | {row[1]:<10} | {row[2]:<10} | {row[3]:<40}")
print("-" * 90)


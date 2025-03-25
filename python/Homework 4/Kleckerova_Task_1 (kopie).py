import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 1. Definice fuzzy proměnných

# Vstupní proměnné
teplota_mistnost = ctrl.Antecedent(np.arange(0, 41, 1), 'teplota_mistnost')
teplota_cilova = ctrl.Antecedent(np.arange(0, 41, 1), 'teplota_cilova')
# Výstupní proměnná
prikaz_klima = ctrl.Consequent(np.arange(-5, 6, 1), 'prikaz_klima')

## 2. Definice fuzzy logiky

# Teplota v místnosti
teplota_mistnost['chladna'] = fuzz.gaussmf(teplota_mistnost.universe, 0, 5)
teplota_mistnost['studena'] = fuzz.gaussmf(teplota_mistnost.universe, 10, 5)
teplota_mistnost['neutralni'] = fuzz.gaussmf(teplota_mistnost.universe, 20, 5)
teplota_mistnost['tepla'] = fuzz.gaussmf(teplota_mistnost.universe, 30, 5)
teplota_mistnost['horka'] = fuzz.gaussmf(teplota_mistnost.universe, 40, 5)

# Cílová teplota
teplota_cilova['chladna'] = fuzz.gaussmf(teplota_cilova.universe, 0, 5)
teplota_cilova['studena'] = fuzz.gaussmf(teplota_cilova.universe, 10, 5)
teplota_cilova['neutralni'] = fuzz.gaussmf(teplota_cilova.universe, 20, 5)
teplota_cilova['tepla'] = fuzz.gaussmf(teplota_cilova.universe, 30, 5)
teplota_cilova['horka'] = fuzz.gaussmf(teplota_cilova.universe, 40, 5)

# Příkaz klimatizace (cílová proměnná)
prikaz_klima['chladit'] = fuzz.gaussmf(prikaz_klima.universe, -5, 1)       
prikaz_klima['zadna_zmena'] = fuzz.gaussmf(prikaz_klima.universe, 0, 0.5) 
prikaz_klima['ohřívat'] = fuzz.gaussmf(prikaz_klima.universe, 5, 1)       

# Vizualizace fuzzy množin
teplota_mistnost.view()
plt.title("Fuzzy množiny - Teplota místnosti")
plt.show()
teplota_cilova.view()
plt.title("Fuzzy množiny - Cílová teplota")
plt.show()
prikaz_klima.view()
plt.title("Fuzzy množiny - Příkaz pro klimatizaci")
plt.show()

# Fuzzy pravidla
pravidlo1 = ctrl.Rule(teplota_mistnost['chladna'] & teplota_cilova['horka'], prikaz_klima['ohřívat'])
pravidlo2 = ctrl.Rule(teplota_mistnost['horka'] & teplota_cilova['chladna'], prikaz_klima['chladit'])
pravidlo3 = ctrl.Rule(teplota_mistnost['studena'] & teplota_cilova['tepla'], prikaz_klima['ohřívat'])
pravidlo4 = ctrl.Rule(teplota_mistnost['tepla'] & teplota_cilova['studena'], prikaz_klima['chladit'])
pravidlo5 = ctrl.Rule(teplota_mistnost['neutralni'] & teplota_cilova['neutralni'], prikaz_klima['zadna_zmena'])
pravidlo6 = ctrl.Rule(teplota_mistnost['tepla'] & teplota_cilova['neutralni'], prikaz_klima['chladit'])
pravidlo7 = ctrl.Rule(teplota_mistnost['studena'] & teplota_cilova['neutralni'], prikaz_klima['ohřívat'])
pravidlo8 = ctrl.Rule(teplota_mistnost['neutralni'] & teplota_cilova['tepla'], prikaz_klima['ohřívat'])
pravidlo9 = ctrl.Rule(teplota_mistnost['neutralni'] & teplota_cilova['studena'], prikaz_klima['chladit'])

## 3. Implementace modelu

klima_ctrl = ctrl.ControlSystem([pravidlo1, pravidlo2, pravidlo3, pravidlo4, pravidlo5, pravidlo6, pravidlo7, pravidlo8, pravidlo9])
klima_sim = ctrl.ControlSystemSimulation(klima_ctrl)

## 4. Testování modelu

def vystup_na_slovo(vystup):
    if vystup <= -0.5:
        return f"Cool ({vystup:.2f})"
    elif vystup > -0.5 and vystup < 0.5:
        return f"No Change ({vystup:.2f})"
    else:
        return f"Heat ({vystup:.2f})"
    
def testovani_modelu(teplota_mistnost, teplota_cilova, ocekavani:str):
    klima_sim.input["teplota_mistnost"] = teplota_mistnost
    klima_sim.input["teplota_cilova"] = teplota_cilova
    klima_sim.compute()
    vystup = klima_sim.output["prikaz_klima"]
    print(f"Teplota místnosti: {teplota_mistnost}°C, Cílová teplota: {teplota_cilova}°C -> Příkaz: {vystup_na_slovo(vystup)} (Očekáváno: {ocekavani})")

# Testování na různých vstupních hodnotách s porovnáním očekávaných výstupů:
print("\nTestování na různých vstupních hodnotách:")
testovani_modelu(10,5,"Cool")
testovani_modelu(5,38,"Heat")
testovani_modelu(5,5, "No change")
testovani_modelu(35,20,"Cool")
testovani_modelu(18,28,"Heat")
testovani_modelu(40,30,"Cool")
testovani_modelu(24,23,"No change")
testovani_modelu(0,40,"Heat")
testovani_modelu(26,20,"Cool")
testovani_modelu(33,30,"Cool")

## 5. Vizualizace výsledků
x = np.arange(0, 41, 1)  # Teplota místnosti (0-40°C)
y = np.arange(0, 41, 1)  # Cílová teplota (0-40°C)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        klima_sim.input["teplota_mistnost"] = X[i, j]
        klima_sim.input["teplota_cilova"] = Y[i, j]
        klima_sim.compute()
        Z[i, j] = klima_sim.output["prikaz_klima"]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
ax.set_xlabel('Teplota místnosti (°C)', fontweight="bold")
ax.set_ylabel('Cílová teplota (°C)', fontweight="bold")
ax.set_zlabel('Příkaz pro klimatizaci', fontweight="bold")
ax.set_title('Závislost mezi teplotou místnosti, cílovou teplotou a příkazem pro klimatizaci', fontweight="bold")
ax.view_init(10,240)
fig.colorbar(surf)
plt.savefig(f"graf_fuzzy_modelovani.png", dpi=300, bbox_inches='tight')
plt.show()


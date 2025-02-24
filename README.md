# KI-PSM
KI/PSM, předmět 2. ročníku letního semestru

## Přednáška 1
# Základy testování hypotéz
- navazuje na kurz **KMA/PAS**
- testuje se platnost tvrzení (př. o normalitě, zda se změnila průměrná výška, ...)
- testují se populační charakteristiky (viz. Inferenční statistika - vyvozování závěrů o celé populaci na základě náhodného výběru z dané populace)
- důležité je správné stanovení **H0**, **HA**
### Chyby testu
1. Chyba 1. druhu ... důležitější, zamítnutí platné nulové hypotézy
2. Chyba 2. druhu ... těžší ji zabránit, přijetí neplatné nulové hypotézy

### p-hodnota
- aktuální dosažená hladina testu
- pravděpodobnost, že za předpokladu platnosti nulové hypotézy nastane daný výsledek, či výsledek horší

### Vybrané druhy testů
1. Testy rozdělení ... (normalita - Shapiro-Wilkův test, ...)
2. Parametrické testy ... (testují hodnotu parametru - t.test, ANNOVA, ... ; PŘEDPOKLAD O ROZDĚLENÍ - většinou normální)
3. Neparametrické testy ... (nevyžadují předpoklad o rozdělení, založené na pořadí, robustní metody - medián; Wilconoxův test)
4. Simulační testy ... (Jen s počítačem, simulace př. 1000 výběrů - permutační test, Bootstrap)

### Schémata testů
viz. Černíková ujep, někde to má

### Dvouvýběrový t test
- srovnávání střední hodnoty 2 nezávislých výběrů
- lze i pomocí intervalového odhadu, svázaného s dvouvýběrovým t.testem
- př. funkce MeanDiffCI dělá základní Welschův test

1. Stanovení H0, HA; H0 ... rozdíl středních hodnot je nulový; HA ... rozdíl není nulový; je statisticky významný **NEBO** H0 ... rozdíl stř. hodnot = nějaké číslo; HA ... nerovná
2. Výběr testu: Dvouvýběrový t test (normální data, rovnost rozptylů)/ Welschův test(normální data, nerovnost rozptylů)/ Wilconoxův test(nenormální data)
3. Test shody dvou rozptylů: H0 ... Rozptyly se neliší, HA ... Rozptyly se liší; vzoreček
4. Provedení testu: vzorečky!



## Přednáška 2
# Významnost
### Vliv velikosti vzorku na výsledek testu
- málo pozorování -> velká p-hodnota
- hodně pozorování -> malá p-hodnota; prokáže se téměř vše
- statistické testy nejlépe fungují s cca **100** pozorováními

### Významnost
- je **věcná** a **statistická**

### Tabulka analýzy rozptylu
- porovnání **vysvětlené** (variabilita mezi výběry - jak se liší průměry) a **nevysvětlené**(jak se v rámci každého výběru liší dané pozorování od stř. hodnoty) variability
- často v **ANOVA** (= porovnání střední hodnoty v nezávislých výběrech)
- více v prezentaci, i cannot :-)













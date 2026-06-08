# Predykcja Przeżywalności Pacjentów z Marskością Wątroby

Projekt zawiera analizę danych i budowę modeli klasyfikacyjnych do przewidywania statusu przeżycia pacjentów z marskością wątroby (Status: **D** – zgon, **C** – cenzurowanie, **CL** – transplantacja wątroby).

## Zbiór danych

Źródłowy plik `cirrhosis.csv` zawiera **418 rekordów** i **20 kolumn**.

| Kolumna         | Opis                                                                                  |
|-----------------|---------------------------------------------------------------------------------------|
| `N_Days`        | Liczba dni obserwacji od rejestracji do śmierci, transplantacji lub zakończenia badań |
| `Status`        | Zmienna docelowa (D = zgon, C = cenzurowanie, CL = transplantacja wątroby)            |
| `Drug`          | Podawany lek (D-penicillamine lub Placebo)                                            |
| `Age`           | Wiek pacjenta w dniach                                                                |
| `Sex`           | Płeć pacjenta (F = kobieta, M = mężczyzna)                                           |
| `Ascites`       | Obecność wodobrzusza (Y/N)                                                            |
| `Hepatomegaly`  | Powiększenie wątroby (Y/N)                                                            |
| `Spiders`       | Naczyniak pajączkowaty (Y/N)                                                          |
| `Edema`         | Obrzęk (N / S / Y)                                                                    |
| `Bilirubin`     | Stężenie bilirubiny w surowicy [mg/dl]                                                |
| `Cholesterol`   | Stężenie cholesterolu w surowicy [mg/dl]                                              |
| `Albumin`       | Stężenie albuminy w surowicy [mg/dl]                                                  |
| `Copper`        | Stężenie miedzi w moczu [ug/day]                                                      |
| `Alk_Phos`      | Aktywność fosfatazy alkalicznej [U/liter]                                             |
| `SGOT`          | Aktywność aminotransferazy asparaginianowej [U/ml]                                    |
| `Tryglicerides` | Stężenie trójglicerydów w surowicy [mg/dl]                                            |
| `Platelets`     | Liczba płytek krwi [ml/1000]                                                          |
| `Prothrombin`   | Czas protrombinowy [s]                                                                |
| `Stage`         | Stadium histologiczne marskości (1–4)                                                 |

## Zawartość notebooka

### 1. Podstawowa analiza i statystyka danych

- Wczytanie danych, sprawdzenie rozmiaru i typów kolumn.
- Obliczenie procentu brakujących wartości per kolumna – kilka kolumn ma ponad 25% wartości NaN.
- Usunięcie kolumny `ID` (brak wartości analitycznej).
- Identyfikacja i usunięcie 15 wierszy, w których brakuje ≥ 50% danych.

### 2. Uzupełnianie brakujących wartości

**Kolumny numeryczne** – decyzja podejmowana na podstawie kształtu rozkładu (histogram ze średnią i medianą):
- Uzupełnienie **medianą** – kolumny o rozkładzie asymetrycznym prawoskośnym: `Cholesterol`, `Copper`, `Alk_Phos`, `Tryglicerides`.
- Uzupełnienie **średnią** – kolumny o rozkładzie zbliżonym do normalnego: `SGOT`, `Platelets`.

**Kolumny kategorialne**:
- `Drug`, `Hepatomegaly` – uzupełnienie wartością **„Unknown"** (rozkład zbliżony 50/50 między istniejącymi wartościami, losowe uzupełnienie mogłoby zakłócić balans klas).
- `Ascites`, `Spiders` – losowe uzupełnienie **zgodnie z rozkładem** istniejących wartości.

### 3. Wykrywanie i obsługa wartości odstających (outlierów)

Dla każdej kolumny numerycznej rysowany jest wykres pudełkowy (boxplot) oraz scatter plot względem zmiennej `Status`, aby ocenić, czy outliery mają wartość predykcyjną.

- Większość wartości odstających **pozostawiona** – korelują ze Statusem `D` (zgon).
- `SGOT` i `Platelets` – ekstremalnie odstające wartości **usunięte** metodą 3×IQR.

### 4. Wizualizacje danych

- Rozkład zmiennej docelowej `Status` (klasa CL jest wyraźnie mniejszościowa ~20 rekordów).
- Heatmapa Stage vs Status.
- Rozkład płci; wpływ płci na odsetek zgonów.
- Wpływ objawów klinicznych (`Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) na śmiertelność – wszystkie cztery cechy wykazują wyraźną korelację z klasą `D`.
- Analiza średnich wartości `Bilirubin` i `Albumin` w zależności od Stage i Status.

### 5. Kodowanie zmiennych kategorialnych

| Kolumna / wartości         | Metoda kodowania                          |
|----------------------------|-------------------------------------------|
| `Sex` (F/M), `Ascites` (Y/N), `Spiders` (Y/N) | Kodowanie binarne (0/1)    |
| `Drug` (3 wartości), `Hepatomegaly` (3 wartości), `Edema` (3 wartości) | One-Hot Encoding (`pd.get_dummies`) |

Po kodowaniu zbiór zawiera **24 cechy** wejściowe.

### 6. Podział danych

Dane podzielono na trzy rozłączne zbiory:
- **Treningowy** – 60%
- **Walidacyjny** – 20%
- **Testowy** – 20%

### 7. Warianty przetwarzania danych

Dla każdego modelu testowano cztery warianty preprocessingu:

| Wariant           | Opis                                                                              |
|-------------------|-----------------------------------------------------------------------------------|
| **Baseline**      | Brak dodatkowego przetwarzania                                                    |
| **Selekcja cech** | `SelectKBest` z testem ANOVA-F, wybór 15 z 24 cech                               |
| **Dyskretyzacja** | `KBinsDiscretizer` (5 równolicznych przedziałów) dla kolumn ciągłych             |
| **PCA**           | `StandardScaler` + redukcja wymiarowości (`PCA`) z 24 do 15–21 komponentów       |

---

## Modele klasyfikacyjne

### Model bazowy – DummyClassifier

Losowy klasyfikator (`strategy="uniform"`) jako punkt odniesienia.
- **Accuracy: ~27,5%** – każdy model poniżej tego progu byłby gorszy od losowego.

---

### Naiwny klasyfikator Bayesa (GaussianNB)

Zakłada niezależność cech i modeluje każdą z nich rozkładem normalnym.

**Strojony hiperparametr:**
- `var_smoothing` – wygładzenie wariancji (testowane wartości: 1e-9 do 0.1). Wyższe wygładzenie obniżało metryki, optymalne okazało się `var_smoothing = 1e-9`.

**Wpływ przetwarzania:** Najlepsze wyniki dało PCA z **n_components = 18**.

| Konfiguracja               | Accuracy | Precision | Recall | F1    |
|----------------------------|----------|-----------|--------|-------|
| GaussianNB Baseline        | 71,25%   | 72,2%     | 71,25% | 71,6% |
| GaussianNB + PCA (n=18)    | **75,0%**| 73,1%     | **75,0%** | 74,0% |

---

### Drzewo decyzyjne (DecisionTreeClassifier)

Buduje hierarchię reguł podziału minimalizującą nieczystość węzłów.

**Strojone hiperparametry** (24 kombinacje):
- `max_depth` ∈ {3, 5, 10, None}
- `criterion` ∈ {`gini`, `entropy`}
- `min_samples_split` ∈ {2, 10, 20}

**Najlepsza konfiguracja:** `max_depth=5`, `criterion=gini`, `min_samples_split=2`

**Analiza przeuczenia (Train vs Test Accuracy):**
- `max_depth=3` – underfitting, Test Accuracy = 68%
- `max_depth=5` – optimum, Test Accuracy = **75%**, gap = 0.13
- `max_depth=10` i `None` – wyraźne przeuczenie (Train ≈ 1.0, Test ≈ 70%)

**Obsługa niezbalansowanych klas:** `class_weight='balanced'` zwiększył Recall dla klasy `CL` (0,25 → 0,50), jednak obniżył ogólne Accuracy do 63,75% ze względu na zbyt małą liczbę próbek tej klasy (~20 rekordów).

| Konfiguracja                      | Accuracy | F1    |
|-----------------------------------|----------|-------|
| DecisionTree Baseline             | 70,0%    | 69,9% |
| DecisionTree Best HP              | **75,0%**| **74,4%** |
| DecisionTree Best HP + Balanced   | 63,75%   | 65,2% |

---

### Las Losowy (RandomForestClassifier)

Ensemble wielu drzew decyzyjnych trenowanych na losowych podzbiorach danych i cech; predykcja przez głosowanie większościowe. Redukuje wariancję i overfitting względem pojedynczego drzewa.

**Strojony hiperparametr:**
- `n_estimators` ∈ {2, 5, 10, 13, 14, 15, 25, 50, 100, 120, 150, 200}

Najlepsze wyniki osiągnięto dla **n_estimators = 14** (Accuracy = **80%**). Przy wyższych wartościach Train Accuracy rośnie do 1.0 (przeuczenie), a Test Accuracy stagnuje na ~77–78%.

Próba mitygacji przeuczenia przez ograniczenie `max_depth`:
- `max_depth=3` – gap spada do 0.06, ale Test Accuracy spada do 75%
- `max_depth=None` – Test Accuracy = **80%** mimo większego gapu

Główna przyczyna przeuczenia: mały zbiór danych (~190 próbek treningowych).

**Wpływ przetwarzania:** Ani selekcja cech, ani `class_weight='balanced'` nie poprawiły wyników – Random Forest jest odporny na monotoniczne transformacje cech i sam ignoruje słabe cechy.

---

## Podsumowanie wyników

| Konfiguracja                      | Accuracy | Precision | Recall | F1    |
|-----------------------------------|----------|-----------|--------|-------|
| DummyClassifier                   | 27,5%    | 39,5%     | 27,5%  | 30,1% |
| GaussianNB Baseline               | 71,25%   | 72,2%     | 71,25% | 71,6% |
| GaussianNB Best HP + PCA (n=18)   | 75,0%    | 73,1%     | 75,0%  | 74,0% |
| DecisionTree Baseline             | 70,0%    | 70,2%     | 70,0%  | 69,9% |
| DecisionTree Best HP              | 75,0%    | 75,1%     | 75,0%  | 74,4% |
| DecisionTree Best HP + Balanced   | 63,75%   | –         | –      | 65,2% |
| **Random Forest Best HP (n=14)**  | **80,0%**| –         | –      | –     |

Najlepszy wynik uzyskał **Random Forest z n_estimators=14** osiągając Accuracy = **80%**. Modele GaussianNB z PCA(18) oraz DecisionTree z dostrojonymi hiperparametrami uzyskały zbliżone wyniki (75%). Klasa `CL` (transplantacja) była trudna do przewidzenia dla wszystkich modeli z powodu bardzo małej liczby próbek (~20 rekordów w całym zbiorze).

## Użyte biblioteki

- `pandas`, `numpy` – przetwarzanie danych
- `matplotlib`, `seaborn` – wizualizacje
- `scikit-learn` – modele ML, metryki, pipeline'y:
  - `DummyClassifier`, `GaussianNB`, `DecisionTreeClassifier`, `RandomForestClassifier`
  - `StandardScaler`, `KBinsDiscretizer`, `SelectKBest`, `PCA`
  - `Pipeline`, `ColumnTransformer`
  - `accuracy_score`, `classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`

# ABSA Sentiment Analysis Dashboard

Aspektowa Analiza Sentymentu (ABSA) dla recenzji lokalizacji z interaktywną wizualizacją na mapie.

## Funkcjonalności

- **Model BERT ABSA**: Fine-tuned BERT do wieloetykietowej klasyfikacji sentymentu aspektowego
- **8 Aspektów**: Safety, Cleanliness, Infrastructure, Nature, Attractions, Heritage, Costs, Other
- **4 Sentymentu**: Positive, Neutral, Negative, Not Mentioned
- **Interaktywna Mapa**: Wizualizacja sentymentu geograficznie z heatmapami i widokami 3D
- **Predykcja na żywo**: Analiza niestandardowych recenzji w czasie rzeczywistym
- **3 Metody predykcji**: Fine-tuned BERT, Zero-shot (BART-MNLI), LLM (GPT-4o-mini)

---

## Wymagania systemowe

| Komponent | Wersja |
|-----------|--------|
| Python | `3.14` |
| uv (package manager) | `>=0.10.0` |
| System | macOS / Linux / Windows |

---

## Instalacja krok po kroku

### 1. Instalacja `uv` (package manager)

`uv` to szybki menedżer pakietów Python. Zainstaluj go jedną z poniższych metod:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Weryfikacja instalacji:**
```bash
uv --version
# Oczekiwany wynik: uv 0.10.x
```

### 2. Instalacja Python 3.14

`uv` automatycznie zarządza wersjami Pythona. Aby zainstalować Python 3.14:

```bash
uv python install 3.14
```

**Weryfikacja:**
```bash
uv python list | grep 3.14
# Powinno pokazać: cpython-3.14.x
```

### 3. Klonowanie repozytorium

```bash
git clone <url-repozytorium>
cd projekt-magisterski
```

### 4. Tworzenie środowiska wirtualnego i instalacja zależności

```bash
# Utworzenie środowiska wirtualnego z Python 3.14
uv venv --python 3.14

# Aktywacja środowiska (macOS/Linux)
source .venv/bin/activate

# Aktywacja środowiska (Windows)
.venv\Scripts\activate

# Instalacja wszystkich zależności z pyproject.toml
uv sync
```

### 5. (Opcjonalnie) Konfiguracja zmiennych środowiskowych

Jeśli chcesz używać predykcji LLM (GPT-4o-mini), utwórz plik `.env`:

```bash
echo "OPENAI_API_KEY=twoj-klucz-api" > .env
```

---

## Uruchomienie aplikacji

### Aplikacja Streamlit (główny dashboard)

```bash
streamlit run app.py
```

Aplikacja uruchomi się domyślnie pod adresem: **http://localhost:8501**

**Dostępne strony w aplikacji:**
- **Home** — główna mapa z wizualizacją sentymentu
- **Data Table** — podgląd danych w formie tabeli
- **Label Dataset** — narzędzie do etykietowania danych
- **Repository** — repozytorium wyników

---

## Notebooki Jupyter

Projekt zawiera notebooki do eksploracji danych, trenowania modelu i ewaluacji. Aby je uruchomić:

```bash
# Uruchomienie Jupyter (jeśli nie masz zainstalowanego)
uv pip install jupyter

# Start Jupyter
jupyter notebook
```

### Główne notebooki

| Notebook | Opis |
|----------|------|
| `_train-absa-model.ipynb` | **Trening modelu ABSA** — fine-tuning BERT na zbiorze etykietowanym. Konfiguracja hiperparametrów, podział train/val, zapis wag modelu. |
| `_model_results.ipynb` | **Porównanie metod** — ewaluacja trzech metod (Fine-tuned BERT, Zero-shot, LLM). Metryki: Precision, Recall, F1, Accuracy. Macierze pomyłek. |
| `_prediction_tests.ipynb` | **Testy predykcji** — szybkie testy predykcji na zbiorze danych dla każdej z trzech metod. |

### Notebooki eksperymentalne (`experiments/`)

| Notebook | Opis |
|----------|------|
| `data-preparation.ipynb` | Przygotowanie i czyszczenie danych |
| `data-labelling-with-ai.ipynb` | Automatyczne etykietowanie danych z AI |
| `eda_reviews.ipynb` | Eksploracyjna analiza recenzji |
| `eda_metadata.ipynb` | Eksploracyjna analiza metadanych |
| `transform-json-to-csv.ipynb` | Konwersja danych JSON do CSV |

### Jak działają notebooki?

1. **Importy z projektu** — Notebooki korzystają z modułów projektu (`config/`, `model/`, `predictions/`), więc muszą być uruchamiane z głównego katalogu projektu.

2. **Konfiguracja globalna** — Wspólna konfiguracja w `config/global_config.py`:
   - `TRAIN_ASPECTS` — lista 8 aspektów do analizy
   - `SENTIMENT_LABELS` — etykiety sentymentu (positive, neutral, negative, notmentioned)
   - `ModelType` — enum z typami modeli (FINE_TUNED, ZERO_SHOT, LLM)
---

## Struktura projektu

```
projekt-magisterski/
├── app.py                      # Główna aplikacja Streamlit
├── pages/                      # Podstrony aplikacji Streamlit
│   ├── home.py                 # Strona główna z mapą
│   ├── data_table.py           # Tabela danych
│   ├── label_dataset.py        # Etykietowanie danych
│   └── repository.py           # Repozytorium wyników
├── config/
│   └── global_config.py        # Globalna konfiguracja (aspekty, etykiety, ścieżki)
├── model/                      # Moduły treningu i predykcji
│   ├── train.py                # Skrypt treningu modelu
│   ├── model.py                # Definicja architektury modelu
│   ├── predict.py              # Funkcje predykcji
│   └── prepare_dataset.py      # Przygotowanie danych do treningu
├── predictions/                # Implementacje metod predykcji
│   ├── predict_dataset.py      # Główna funkcja predykcji
│   ├── prediction_fine_tuned.py # Predykcja z fine-tuned BERT
│   ├── prediction_zero_shot.py  # Predykcja zero-shot
│   └── prediction_llm.py       # Predykcja z LLM
├── application/                # Komponenty aplikacji
│   ├── map_components.py       # Komponenty mapy
│   ├── visual_settings.py      # Ustawienia wizualne
│   └── ...
├── statics/
│   ├── datasets/               # Zbiory danych
│   ├── models/                 # Zapisane wagi modeli
│   └── results_repository/     # Wyniki eksperymentów
├── experiments/                # Notebooki eksperymentalne
├── _train-absa-model.ipynb     # Notebook treningu
├── _model_results.ipynb        # Notebook ewaluacji
├── _prediction_tests.ipynb     # Notebook testów
├── pyproject.toml              # Konfiguracja projektu i zależności
└── .python-version             # Wersja Pythona (3.14)
```

---

## Zależności (zablokowane wersje)

Wszystkie zależności są zdefiniowane w `pyproject.toml`:

| Pakiet | Wersja |
|--------|--------|
| torch | `>=2.11.0` |
| transformers | `>=5.4.0` |
| streamlit | `>=1.45.0` |
| pandas | `>=3.0.1` |
| polars | `>=1.18.0` |
| scikit-learn | `>=1.8.0` |
| openai | `>=2.30.0` |
| spacy | `>=3.8.13` |
| pydeck | `>=0.9.0` |
| folium | `>=0.20.0` |
| matplotlib | `latest` |
| pydantic | `>=2.12.5` |
| tqdm | `>=4.67.3` |
| emoji | `>=2.15.0` |
| playwright | `>=1.58.0` |
| nbconvert | `>=7.17.0` |

---

## Rozwiązywanie problemów

### Python 3.14 nie jest dostępny
```bash
uv python install 3.14
```

### Błąd importu modułów projektu w notebookach
Upewnij się, że uruchamiasz notebook z głównego katalogu projektu:
```bash
cd projekt-magisterski
jupyter notebook
```

### Błąd CUDA / MPS
Projekt automatycznie wykrywa dostępne urządzenie (CUDA, MPS, CPU). Jeśli masz problemy z GPU, 
w większości kodu urządzenie jest dobierane automatycznie i zostalo to przetestowane na urządzeniu MacBook Pro m4.

```python
import torch
print(torch.backends.mps.is_available())  # macOS
print(torch.cuda.is_available())          # NVIDIA
```

### Błąd OpenAI API
Upewnij się, że masz ustawiony klucz API:
```bash
export OPENAI_API_KEY="twoj-klucz"
```

---

## Licencja

Projekt realizowany w ramach pracy magisterskiej.

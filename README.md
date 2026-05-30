# ABSA — aspektowa analiza sentymentu recenzji lokalizacji

Aplikacja webowa (Streamlit) do aspektowej analizy sentymentu (ABSA) recenzji miejsc turystycznych z wizualizacją wyników na interaktywnej mapie.

Projekt jest dystrybuowany jako **archiwum ZIP**. Po rozpakowaniu wykonaj kroki instalacji opisane poniżej — **nie jest wymagany Git ani konto GitHub**.

**8 aspektów:** safety, cleanliness, infrastructure, nature, attractions, heritage, costs, other  
**4 etykiety sentymentu:** positive, neutral, negative, notmentioned

---

## Szybki start

Poniższe kroki zakładają, że archiwum ZIP projektu jest **już rozpakowane** na dysku lokalnym.

```bash
# 1. Wejdź do katalogu projektu (dostosuj ścieżkę do swojej lokalizacji)
cd /ścieżka/do/projekt-magisterski

# 2. Zainstaluj uv (jeśli nie masz) — patrz sekcja „Wymagane narzędzia”
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Zainstaluj Pythona 3.14 i utwórz środowisko wirtualne
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 4. Zainstaluj zależności projektu
uv sync

# 5. Uruchom aplikację
uv run streamlit run app.py
```

Aplikacja otworzy się pod adresem **http://localhost:8501**.

Model **TF-IDF + LSA** działa od razu po instalacji (trenuje się automatycznie przy pierwszym użyciu z pliku `statics/datasets/training.csv`). Modele **BERT / DistilBERT** wymagają wcześniejszego wytrenowania — patrz sekcja [Trening modeli](#trening-modeli).

---

## Wymagania systemowe

| Komponent | Wersja | Link |
|-----------|--------|------|
| Python | `3.14` | [python.org/downloads](https://www.python.org/downloads/) |
| uv (menedżer pakietów) | `>= 0.10.0` | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |
| System | macOS / Linux / Windows | — |
| RAM | min. 16 GB (zalecane 24 GB przy modelach BERT) | — |
| Dysk | ~5 GB (PyTorch, transformers, cache Hugging Face) | — |

---

## Wymagane narzędzia (instalacja jednorazowa)

Przed pierwszym uruchomieniem projektu zainstaluj poniższe narzędzia. Nie są one dołączone do archiwum ZIP.

### `uv` — menedżer pakietów Python

- Dokumentacja: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- Instalator: [https://astral.sh/uv](https://astral.sh/uv)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Weryfikacja:**
```bash
uv --version
```

### Python 3.14

`uv` może pobrać i zarządzać wersją Pythona automatycznie:

```bash
uv python install 3.14
uv python list | grep 3.14
```

Alternatywnie Python można zainstalować ręcznie ze strony [https://www.python.org/downloads/](https://www.python.org/downloads/) (wersja 3.14).

---

## Instalacja krok po kroku
> **Uwaga:** Wszystkie poniższe komendy należy wykonywać z katalogu głównego projektu.

### 1. Instalacja `uv` i Pythona 3.14

Jeśli nie masz jeszcze zainstalowanych narzędzi, wykonaj kroki z sekcji [Wymagane narzędzia](#wymagane-narzędzia-instalacja-jednorazowa).

### 2. Środowisko wirtualne i zależności

```bash
uv venv --python 3.14
source .venv/bin/activate        # macOS / Linux

uv sync
```

Polecenie `uv sync` instaluje wszystkie biblioteki wymienione w `pyproject.toml` (PyTorch, Streamlit, transformers itd.) zgodnie z wersjami zablokowanymi w pliku `uv.lock`.

**Przy pierwszej instalacji pobierane są duże pakiety** (PyTorch ~2 GB, modele Hugging Face przy treningu). Wymagane jest połączenie z internetem.


---

## Uruchomienie aplikacji

Upewnij się, że jesteś w katalogu projektu i masz aktywne środowisko wirtualne:

```bash
cd /ścieżka/do/projekt-magisterski
source .venv/bin/activate 

uv run streamlit run app.py
```

Domyślny adres: **http://localhost:8501**  
Dokumentacja Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)

### Strony aplikacji

| Strona | Plik | Opis |
|--------|------|------|
| Strona główna | `pages/home.py` | Mapa z wizualizacją sentymentu per aspekt |
| Tabela danych | `pages/data_table.py` | Podgląd zbiorów w formie tabeli |
| Etykietowanie danych | `pages/label_dataset.py` | Automatyczne etykietowanie zbiorów modelem ML |
| Repozytorium | `pages/repository.py` | Upload, podgląd i usuwanie zbiorów danych |

---

## Pierwsze kroki w aplikacji

Typowy przepływ pracy:

1. **Repozytorium** — wgraj plik CSV z recenzjami (typ: *Surowe — opinie* lub *Oczyszczone*). Wymagane kolumny m.in.: `name`, `latitude`, `longitude`, `text`, `time`, `rating`.
2. **Etykietowanie danych** — wybierz zbiór źródłowy i model (np. TF-IDF + LSA), uruchom etykietowanie, zapisz wynik do repozytorium jako `LABELLED_AI`.
3. **Strona główna** — wybierz etykietowany zbiór, aspekt i przeglądaj wyniki na mapie. Kliknięcie punktu otwiera szczegóły miejsca i listę opinii.

Przykładowe pliki CSV do testów znajdują się w `statics/datasets/`:
- `training.csv` — zbiór treningowy (używany przez TF-IDF)
- `validate.csv` — zbiór walidacyjny
- `tfidf.csv` — dane do eksperymentów TF-IDF

Repozytorium użytkownika (`statics/results_repository/`) tworzy się automatycznie przy pierwszym zapisie.

---

## Modele predykcji

W interfejsie aplikacji dostępne są trzy modele:

| Model w UI | Typ w kodzie | Wymagania |
|------------|--------------|-----------|
| TF-IDF + LSA + regresja logistyczna | `ModelType.TFIDF_LSA` | Działa od razu — trenuje się lazy z `statics/datasets/training.csv` |
| BERT (fine-tuned) | `ModelType.FINE_TUNED_BERT` | Plik `saved_models/bert-base-uncased_absa.pt` |
| DistilBERT (fine-tuned) | `ModelType.FINE_TUNED_DISTILBERT` | Plik `saved_models/distilbert-base-uncased_absa.pt` |

Dodatkowe modele zdefiniowane w kodzie (`predictions/predict_dataset.py`), używane głównie w notebookach:

| Typ | Plik checkpointu |
|-----|------------------|
| `TFIDF_LSA_RF` | — (trenuje się lazy, Random Forest zamiast LR) |
| `FINE_TUNED_DISTILBERT_SST` | `saved_models/distilbert-base-uncased-finetuned-sst-2-english_absa.pt` |
| `TEST_BERT_BASE_UNCASED_ABSA` | `saved_models/distilbert-base-uncased-finetuned-sst-2-english_test_absa.pt` |

Katalog `saved_models/` może być pusty w archiwum ZIP — checkpointy powstają po treningu (patrz poniżej) lub mogą być dołączone osobno przez autora projektu.

---

## Trening modeli

Trening odbywa się przez notebooki Jupyter uruchamiane z **głównego katalogu projektu**:

- Jupyter: [https://jupyter.org/](https://jupyter.org/)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)

```bash
cd /ścieżka/do/projekt-magisterski
source .venv/bin/activate

uv pip install jupyter
jupyter notebook
```

| Notebook | Opis |
|----------|------|
| `_train_absa_model.ipynb` | Fine-tuning BERT (`bert-base-uncased`) → `saved_models/bert-base-uncased_absa.pt`. Zawiera wykres historii treningu (loss, F1). |
| `_train_absa_model-distilbert.ipynb` | Fine-tuning DistilBERT → `saved_models/distilbert-base-uncased_absa.pt`. Zawiera wykres historii treningu. |
| `_model_results.ipynb` | **Główny notebook wyników badań** — porównanie metod predykcji, metryki F1, macierze pomyłek, macierz zgodności między modelami. |
| `_tf_idf_results.ipynb` | Wyniki baseline TF-IDF + LSA — ewaluacja i porównanie z modelami neuronowymi. |

Notebooki eksperymentalne w katalogu `experiments/`:
- `data-preparation.ipynb` — przygotowanie i czyszczenie danych
- `eda_reviews.ipynb`, `eda_metadata.ipynb` — analiza eksploracyjna
- `transform-json-to-csv.ipynb` — konwersja JSON → CSV

---

## Wyniki badań

Wyniki badań magisterskich znajdują się w **notebookach Jupyter** w katalogu głównym projektu:

| Notebook | Co zawiera |
|----------|--------------|
| `_model_results.ipynb` | Najważniejszy plik. Porównanie modeli, metryki per aspekt, macierze pomyłek, macierz zgodności |
| `_train_absa_model.ipynb` | Proces treningu BERT, krzywe uczenia, wyniki walidacji |
| `_train_absa_model-distilbert.ipynb` | Proces treningu DistilBERT, krzywe uczenia, wyniki walidacji |
| `_tf_idf_results.ipynb` | Wyniki baseline TF-IDF + LSA |

Po uruchomieniu notebooków (lub skryptu `wyniki_badań/generate_results.py`) wygenerowane pliki trafiają do katalogu **`wyniki_badań/`**:

| Plik | Opis |
|------|------|
| `table5_model_comparison.csv` | Tabela porównawcza modeli (Tabela 5 z pracy) |
| `per_aspect_f1_4class.csv` / `per_aspect_f1_mentioned.csv` | F1 per aspekt |
| `per_aspect_f1_heatmap.png` | Mapa ciepła F1 per aspekt |
| `confusion_matrices.png` | Macierze pomyłek |
| `agreement_matrix.csv` / `.png` | Macierz zgodności między metodami |
| `coverage_by_aspect.csv` / `.png` | Pokrycie aspektów w danych |
| `training_history.png` | Historia treningu (loss, F1) |
| `dataset_statistics.json` | Statystyki zbioru danych |
| `test_results.txt` | Wyniki testów automatycznych (`pytest`) |

---

## Testy automatyczne

- pytest: [https://docs.pytest.org/](https://docs.pytest.org/)

```bash
cd /ścieżka/do/projekt-magisterski
source .venv/bin/activate

uv run pytest
```

Testy obejmują m.in. walidację uploadu CSV, logikę mapy, repozytorium i predykcję. Scenariusze testów manualnych aplikacji opisane są w `tests/manual_tests.md`.

---

## Struktura projektu

```
projekt-magisterski/
├── app.py                          # Punkt wejścia Streamlit
├── pages/                          # Strony aplikacji
├── config/global_config.py         # Aspekty, etykiety, ścieżki, enum ModelType
├── model/                          # Architektura, trening, predykcja dual-head
├── predictions/                    # Implementacje modeli (BERT, TF-IDF, LLM)
├── application/                    # Logika UI: mapa, repozytorium, walidacja
├── statics/
│   ├── datasets/                   # Przykładowe CSV (training, validate, tfidf)
│   ├── icons/                      # Ikony aplikacji
│   └── results_repository/         # Repozytorium użytkownika (tworzone runtime)
├── saved_models/                   # Checkpointy BERT/DistilBERT (po treningu)
├── wyniki_badań/                   # Wygenerowane wyniki badań (CSV, PNG, JSON)
├── _model_results.ipynb            # Notebook z wynikami porównania modeli
├── _train_absa_model.ipynb         # Notebook treningu BERT + wyniki uczenia
├── _train_absa_model-distilbert.ipynb  # Notebook treningu DistilBERT + wyniki uczenia
├── _tf_idf_results.ipynb           # Notebook wyników baseline TF-IDF
├── experiments/                    # Notebooki EDA i przygotowania danych
├── tests/                          # Testy pytest + manual_tests.md
├── pyproject.toml                  # Zależności projektu
└── uv.lock                         # Zablokowane wersje pakietów
```

---

## Konfiguracja

Wspólne stałe w `config/global_config.py`:

- `TRAIN_ASPECTS` — lista 8 aspektów
- `SentimentLabel` / `SENTIMENT_LABELS` — etykiety sentymentu
- `ModelType` — dostępne typy modeli
- `MAX_LENGTH = 128` — maksymalna długość tokenów
- `RESULTS_REPOSITORY_DIR` — katalog repozytorium (`statics/results_repository`)
- `MODEL_DIR` — domyślny katalog modeli (`statics/models`)

---

## Rozwiązywanie problemów

### Python 3.14 nie jest dostępny
```bash
uv python install 3.14
```

### Błąd importu modułów w notebookach
Uruchamiaj Jupyter z głównego katalogu projektu (tam, gdzie leży `app.py`):
```bash
cd /ścieżka/do/projekt-magisterski
source .venv/bin/activate
jupyter notebook
```

### `Checkpoint not found` przy modelu BERT/DistilBERT
Wytrenuj model notebookiem `_train_absa_model.ipynb` lub `_train_absa_model-distilbert.ipynb`, albo użyj modelu TF-IDF + LSA, który nie wymaga checkpointu.

### Wolne pierwsze etykietowanie TF-IDF
Przy pierwszym uruchomieniu model TF-IDF trenuje się na `statics/datasets/training.csv` — to normalne i trwa kilkadziesiąt sekund.

### CUDA / MPS / CPU
Projekt automatycznie korzysta z dostępnego urządzenia PyTorch (CUDA, Apple MPS lub CPU). Przetestowano na MacBook Pro M4.

```python
import torch
print(torch.backends.mps.is_available())  # macOS Apple Silicon
print(torch.cuda.is_available())          # NVIDIA GPU
```

### Port 8501 zajęty
Streamlit wybierze kolejny wolny port (np. 8502). 
```

---

## Przydatne linki

| Narzędzie / biblioteka | Adres |
|------------------------|-------|
| uv (menedżer pakietów) | [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) |
| Instalator uv | [https://astral.sh/uv](https://astral.sh/uv) |
| Python 3.14 | [https://www.python.org/downloads/](https://www.python.org/downloads/) |
| Streamlit | [https://docs.streamlit.io/](https://docs.streamlit.io/) |
| PyTorch | [https://pytorch.org/](https://pytorch.org/) |
| Hugging Face Transformers | [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/) |
| Jupyter Notebook | [https://jupyter.org/](https://jupyter.org/) |
| pytest | [https://docs.pytest.org/](https://docs.pytest.org/) |

---

## Licencja

Projekt realizowany w ramach pracy magisterskiej.

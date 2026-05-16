# Raport testów regresyjnych — testy manualne

Niniejszy dokument zawiera ustandaryzowane scenariusze testów manualnych, opracowane
w celu zapewnienia powtarzalności weryfikacji kluczowych funkcjonalności aplikacji.
Przed każdym wdrożeniem nowej wersji systemu należy przeprowadzić poniższe scenariusze
i uzupełnić kolumnę **Status**.

**Legenda statusów:** PASS | FAIL | SKIP | N/T (nie testowano)

---

## Scenariusz 1 — Upload i walidacja pliku CSV do repozytorium

**Ścieżka:** zarządzanie repozytorium danych

| Pole              | Wartość                                                    |
|-------------------|------------------------------------------------------------|
| **Warunki wstępne** | Aplikacja uruchomiona (`streamlit run app.py`). Repozytorium zawiera co najmniej jeden wpis lub jest puste. Przygotowany poprawny plik CSV z kolumnami: `name`, `latitude`, `longitude`, `text`, `time`, `rating`. |

| Krok | Opis                                                                                     | Oczekiwany rezultat                                                               | Status |
|-----:|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------|
|    1 | Otwórz stronę **Repozytorium** z menu nawigacyjnego.                                    | Wyświetla się lista istniejących zbiorów danych (lub informacja o pustym repozytorium). |        |
|    2 | W sekcji uploadu wybierz typ zbioru: **Surowe — opinie** (`RAW_REVIEWS`).               | Formularz uploadu jest widoczny, pole typu zbioru ustawione na wybraną wartość.    |        |
|    3 | Wgraj przygotowany plik CSV za pomocą komponentu uploadu plików.                        | Plik zostaje zaakceptowany, brak komunikatu o błędzie walidacji.                   |        |
|    4 | Sprawdź, czy nowy wpis pojawił się na liście zbiorów danych.                            | Wpis widoczny na liście z poprawną nazwą pliku i typem `Surowe — opinie`.          |        |
|    5 | Spróbuj wgrać plik z brakującą kolumną `latitude`.                                      | System wyświetla komunikat błędu: *„Dataset does not contain the required column: latitude"*. |        |
|    6 | Spróbuj wgrać pusty plik (0 bajtów).                                                    | System wyświetla komunikat: *„No file uploaded."*                                  |        |

---

## Scenariusz 2 — Etykietowanie zbioru danych modelem AI

**Ścieżka:** etykietowanie zbiorów danych z wykorzystaniem modeli sztucznej inteligencji

| Pole              | Wartość                                                    |
|-------------------|------------------------------------------------------------|
| **Warunki wstępne** | W repozytorium znajduje się co najmniej jeden zbiór typu `RAW_REVIEWS` lub `CLEANED`. Modele predykcyjne są dostępne (np. TF-IDF + LSA). |

| Krok | Opis                                                                                     | Oczekiwany rezultat                                                               | Status |
|-----:|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------|
|    1 | Otwórz stronę **Etykietowanie zbioru danych** z menu nawigacyjnego.                     | Wyświetla się formularz wyboru modelu i zbioru źródłowego.                         |        |
|    2 | Wybierz model: **TF-IDF + LSA + regresja logistyczna**.                                 | Model zaznaczony w polu wyboru.                                                    |        |
|    3 | Wybierz zbiór źródłowy z rozwijanej listy.                                              | Nazwa pliku widoczna, pod spodem informacja o liczbie wierszy.                     |        |
|    4 | Ustaw liczbę wierszy do załadowania (np. 50) i kliknij **Uruchom etykietowanie**.       | Pasek postępu pojawia się i stopniowo wypełnia od 0% do 100%.                     |        |
|    5 | Po zakończeniu etykietowania sprawdź podgląd wyników.                                   | Tabela z kolumnami aspektowymi (`safety`, `cleanliness`, …) wypełnionymi etykietami sentymentu. |        |
|    6 | Wpisz nazwę pliku wyjściowego (np. `wynik_labelled.csv`) i kliknij **Zapisz do repozytorium**. | Komunikat sukcesu: *„Zapisano wynik_labelled.csv jako LABELLED_AI"*.              |        |
|    7 | Przejdź do strony Repozytorium i zweryfikuj, że nowy wpis istnieje z typem `LABELLED_AI`. | Wpis widoczny na liście z prawidłowym typem i notatką o źródle.                   |        |

---

## Scenariusz 3 — Wyświetlanie wyników analizy na mapie interaktywnej

**Ścieżka:** wyświetlanie wyników analizy na mapie przestrzennej

| Pole              | Wartość                                                    |
|-------------------|------------------------------------------------------------|
| **Warunki wstępne** | W repozytorium znajduje się co najmniej jeden zbiór typu `LABELLED_AI` lub `LABELLED_HUMAN` z poprawnymi współrzędnymi geograficznymi. |

| Krok | Opis                                                                                     | Oczekiwany rezultat                                                               | Status |
|-----:|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------|
|    1 | Otwórz stronę główną (Home) aplikacji.                                                  | Wyświetla się selektor zbioru danych oraz mapa w ciemnym stylu.                    |        |
|    2 | Wybierz zbiór danych typu `LABELLED_AI` z listy rozwijanej.                             | Mapa ładuje się z punktami odpowiadającymi lokalizacjom z wybranego zbioru.        |        |
|    3 | Wybierz aspekt (np. `safety`) z selektora aspektów.                                     | Punkty na mapie zmieniają kolor odpowiadający rozkładowi sentymentu dla wybranego aspektu. |        |
|    4 | Najedź kursorem na punkt na mapie.                                                      | Wyświetla się tooltip z nazwą miejsca (`{place_name}`) oraz informacją *„Kliknij dla szczegółów"*. |        |
|    5 | Zweryfikuj, że punkty z przewagą pozytywnych etykiet mają kolor zielony, a negatywnych — czerwony. | Kolory punktów odpowiadają dominującemu sentymentowi: zielony = pozytywny, czerwony = negatywny. |        |
|    6 | Sprawdź, czy wiele opinii o tym samym `gmap_id` jest agregowanych w jeden punkt.        | Jeden punkt na mapie z promieniem proporcjonalnym do liczby opinii (tooltip: `n_reviews`). |        |

---

## Scenariusz 4 — Podgląd szczegółowych informacji o obiekcie

**Ścieżka:** podgląd szczegółowych informacji o obiektach

| Pole              | Wartość                                                    |
|-------------------|------------------------------------------------------------|
| **Warunki wstępne** | Mapa załadowana ze zbiorem `LABELLED_AI`/`LABELLED_HUMAN`. Na mapie widoczne są punkty. |

| Krok | Opis                                                                                     | Oczekiwany rezultat                                                               | Status |
|-----:|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------|
|    1 | Kliknij na punkt (marker) na mapie.                                                     | Otwiera się panel szczegółów z informacjami o wybranym miejscu.                    |        |
|    2 | Zweryfikuj, że panel zawiera nazwę miejsca.                                             | Nazwa miejsca (`name`) jest widoczna w nagłówku panelu.                            |        |
|    3 | Sprawdź, czy wyświetlane są kategorie miejsca (jeśli dostępne).                         | Kategoria (`category`) jest wyświetlona pod nazwą (np. *„Museum\|\|Art"*).         |        |
|    4 | Zweryfikuj listę opinii przypisanych do wybranego miejsca.                              | Wyświetlone są teksty opinii (`text`) powiązane z danym `_place_key`.              |        |
|    5 | Sprawdź zliczenia sentymentu dla wybranego aspektu (np. `safety`).                      | Widoczne są liczby opinii dla każdej etykiety: positive, neutral, negative, notmentioned. |        |
|    6 | Wybierz filtr sentymentu (np. *positive*) i zweryfikuj, że lista opinii się zawęża.     | Wyświetlone są tylko opinie z etykietą `positive` dla wybranego aspektu.           |        |

---

## Scenariusz 5 — Usuwanie zbioru danych z repozytorium

**Ścieżka:** zarządzanie repozytorium danych

| Pole              | Wartość                                                    |
|-------------------|------------------------------------------------------------|
| **Warunki wstępne** | W repozytorium znajdują się co najmniej dwa wpisy. Zanotuj nazwy plików przed testem. |

| Krok | Opis                                                                                     | Oczekiwany rezultat                                                               | Status |
|-----:|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------|
|    1 | Otwórz stronę **Repozytorium**.                                                        | Lista zbiorów danych wyświetla oba wpisy.                                          |        |
|    2 | Wybierz jeden ze zbiorów danych i użyj opcji usuwania.                                  | System prosi o potwierdzenie usunięcia (lub usuwa natychmiast, zależnie od UI).    |        |
|    3 | Potwierdź usunięcie.                                                                    | Komunikat sukcesu. Usunięty wpis znika z listy zbiorów.                            |        |
|    4 | Odśwież stronę (F5) i sprawdź, czy wpis nadal nie jest widoczny.                        | Lista zawiera tylko pozostały wpis. Usunięty plik nie powrócił.                    |        |
|    5 | Przejdź na stronę główną (Home) i sprawdź, czy usunięty zbiór nie jest dostępny w selektorze. | Usunięty zbiór nie pojawia się w rozwijanej liście zbiorów danych na mapie.        |        |

---

## Podsumowanie wykonania

| Scenariusz | Nazwa                                           | Wynik  | Data       | Tester |
|-----------:|-------------------------------------------------|--------|------------|--------|
|          1 | Upload i walidacja pliku CSV do repozytorium    |        |            |        |
|          2 | Etykietowanie zbioru danych modelem AI          |        |            |        |
|          3 | Wyświetlanie wyników na mapie interaktywnej     |        |            |        |
|          4 | Podgląd szczegółowych informacji o obiekcie     |        |            |        |
|          5 | Usuwanie zbioru danych z repozytorium           |        |            |        |

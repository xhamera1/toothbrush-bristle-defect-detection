# Raport: wykrywanie wad szczoteczki (segmentacja binarna)

**Projekt:** segmentacja pikselowa usterek areału włosia / główki szczoteczki  
**Kontekst:** kurs Zaawansowane algorytmy wizyjne (ZAW), AGH  
**Ewaluacja:** średnia **IoU** (Intersection over Union) między maską predykcji a maską referencyjną; wdrożenie konkursowe **CodaBench**

---

## 1. Cel i zakres

Celem jest **maska binarna** \(H \times W\), `uint8`, wartości **0** lub **255**, wskazująca obszary wady na zdjęciu szczoteczki (RGB, typowo **1024×1024**, ciemne tło). Wady obejmują m.in. rozchylone włosie, ubytki / nieregularności wewnątrz kołnierza włosia oraz trudne przypadki „**białe na białym**” (np. zadziory, klejące się kępki).

Podejście jest **hybrydowe**: deterministyczna wizja klasyczna (OpenCV) + **U-Net** (głębokie uczenie) na wyciętym ROI, z późniejszą **ewolucją inferencji** (TTA, skale, próg, post-processing), bez zmiany wag na etapie zgłoszenia.

---

## 2. Charakterystyka danych

| Aspekt | Opis |
|--------|------|
| Rozdzielczość | Zwykle 1024×1024, RGB |
| Tło | Ciemne — obiekt stanowi część kadru |
| Zestaw treningowy (opis metodyczny) | ~30 obrazów **defective**, ~60 **good** (+ maski dla wadliwych) |
| Maska GT | Binarna: 0 = brak wady, 255 = wada |
| Trudność | Małe i kontrastowe wady łatwiejsze dla sieci; wady jasne na jasnym plastiku/włosiu wymagają lepszego nadzoru w treningu lub delikatnych heurystyk (agresywne reguły koloru tekstury psują obrazy „good”) |

Źródło zbiorów opartych o scenariusze przemysłowe: często cytowany zestaw **MVTec AD** (referencja literaturowa w materiałach kursu/konkursu).

---

## 3. Dwa notebooki — przebieg, sens kolejnych kroków

W workflow projektu wyróżniamy **dwa główne notatniki Jupyter** (nazwy mogą się minimalnie różnić w zależności od wersji repozytorium; poniżej opis **logiczny** zgodny z przyjętą metodologią).

### 3.1. Notatnik 1 — przetwarzanie wstępne i przygotowanie zbioru (`01_cv_preprocessing.ipynb` lub równorzędny)

**Cel:** ograniczyć czarny margines, **wyciąć ROI** szczoteczki i zbudować **zbiór treningowy z wycinków** (obraz + maska), żeby U-Net uczył się na rozmiarze **256×256** na treściwym fragmencie obiektu, a nie na pustym tle.

**Kolejność typowych komórek i po co:**

1. **Importy i ścieżki** — wczytanie list plików z katalogów `defective` / `good` oraz `ground_truth`.
2. **Wczytanie obrazu (BGR→RGB)** i **maski** (skala szarości).
3. **Segmentacja „ciała” szczoteczki (klasyka)** — np. HSV, kanał jasności, ewentualnie CLAHE, rozmycie, progowanie (Otsu + korekta), morfologia, wybór największego konturu i wypełnienie.  
   **Po co:** stabilny **bounding box** obiektu niezależnie od małej wady na włosiu.
4. **Padding ROI** — rozszerzenie prostokąta o margines (np. kilkadziesiąt pikseli), żeby nie obcinać rzadkich włókien na krawędzi.  
   **Po co:** spójność z inferencją i mniejsze **obcięcie** etykiet.
5. **Wycięcie** `crop_rgb` i `crop_mask` do tych samych współrzędnych.  
   **Po co:** trening na mniejszym, sensownym patchu ≈ **większa efektywna rozdzielczość wady** po resize do 256².
6. **Zapis lub buforowanie** — zapis par (obraz, maska) do struktury katalogów lub tensorów pod DataLoader.  
   **Po co:** drugi notatnik i pętla treningowa czytają już **gotowy**, spójny zbiór.

**Idea główna:** sieć nie musi „gasić” całego 1024²; uczy się wzorca włosia i anomalii w obręcie ROI — to typowa optymalizacja pod mały zbiór i CPU.

---

### 3.2. Notatnik 2 — trening U-Net (`02_unet_training.ipynb` lub równorzędny)

**Cel:** nauczyć model **`segmentation_models_pytorch` + U-Net + encoder ResNet34** (lub analogiczna konfiguracja) przewidywać maskę wady na wycinku 256×256, z metryką zbliżoną do IoU używanej w ewaluacji.

**Kolejność typowych komórek i po co:**

1. **Konfiguracja** — `seed`, ścieżki do cropów, rozmiar wejścia 256, batch (np. 8), liczba epok (np. ~40 na CPU).
2. **Dataset / DataLoader** — para (obraz, maska); maska binarna lub {0,1}.  
   **Po co:** deterministyczne partie pod uczenie i walidację.
3. **Augmentacje (trening)** — np. odbicia poziome/pionowe, delikatna zmiana jasności/kontrastu (**Albumentations**).  
   **Po co:** generalizacja przy **ograniczonej liczbie** zdjęć; walidacja zwykle bez agresywnych augmentacji.
4. **Model** — `smp.Unet`, encoder ResNet34, często pretrening ImageNet **tylko na etapie uczenia**; w zgłoszeniu ładujemy **własny** `weights.pth` z `encoder_weights=None`.
5. **Funkcja straty** — np. **Dice** (binary) lub kombinacja z BCE — dobrą praktyką przy niezbalansowanym tle (wada = mało pikseli).  
   **Po co:** stabilniejszy gradient niż samo BCE na rzadkiej masce.
6. **Pętla treningowa** — forward, loss, backward, śledzenie **train IoU / loss** i **val IoU / loss** co epokę.
7. **Zapis checkpointu** — najlepszy model po walidacji lub ostatnia epoka → eksport do `weights.pth` używanego w `model.py`.  
   **Po co:** jeden plik wag pod Docker/CodaBench bez internetu.

**Idea główna:** U-Net „widzi” detale wewnątrz kołnierza włosia; jakość na **białych** wadach zależy od pokrycia tego typu przykładami w zbiorze i od augmentacji — sama inferencja bez dobrego treningu ma sufit.

---

## 3.3. Uwaga o repozytorium

W bieżącym drzewie projektu pliki **`.ipynb`** mogą nie być dołączone (np. tylko lokalnie na dysku zajęciowym). Opis powyżej odzwierciedla **założony, spójny pipeline** opisany wcześniej w dokumentacji projektu i wymagany do powstania `weights.pth` oraz logicznej zgodności z kodem inferencji (ROI 256, normalizacja ImageNet).

---

## 4. Model inferencyjny (`model.py`) — architektura i sens modułów

Plik przeznaczony do zgłoszenia (np. katalog `submission5/`) implementuje klasę **`ToothbrushDefectDetector`** i funkcję **`predict(image)`** zgodnie z API CodaBench.

### 4.1. Środowisko i trwałość zgłoszenia

- **`SCRIPT_DIR`** + `weights.pth` obok `model.py` — **względna ścieżka do pliku skryptu**, nie do bieżącego katalogu roboczego.  
  **Po co:** w kontenerze CodaBench `cwd` może być inny — uniknięcie „nie znaleziono wag” i wyniku 0.
- **`opencv-python-headless`** — brak GUI, stabilność w Dockerze.
- Inicjalizacja detektora **raz** przy imporcie modułu — mniej narzutu przy wielu obrazach.

### 4.2. Etapy predykcji (kolejność w `predict`)

1. **Maska „ciała”** (`_get_body_mask`) — HSV, CLAHE na V, rozmycie, Otsu + próg złożony, Canny + delikatna dilatacja, morfologia zamknięcie/otwarcie, **największy kontur** wypełniony.  
   **Po co:** jeden spójny obszar szczoteczki; dalsze kroki ograniczone do obiektu.

2. **Wady zewnętrzne** (`_get_external_defects`) — opening na masce ciała, „rdzeń”, dylatacja marginesu, różnica: obszary maski ciała poza rozszerzonym rdzeniem ≈ **rozchylone włosie**.  
   **Po co:** deterministyczny sygnał geometryczny bez sieci.

3. **Ciemniejsze wewnętrzne** (`_get_internal_dark_defects`) — V-channel, Otsu, ciemniejsza połowa odwrócona, erozja maski ciała (bezpieczna strefa), AND z ciemnymi pikselami, otwarcie szumu.  
   **Po co:** uzupełnienie U-Netu tam, gdzie wada obniża jasność (inny kanał niż sama sieć).

4. **U-Net na ROI** — prostokąt z konturu + padding; wycinek RGB; **`_predict_with_tta`**:
   - **Odbicia** w płaszczyźnie obrazu (oryginał, poziomo, pionowo, obie),
   - **Wieloskalowość ROI** (`ROI_TTA_SCALES = (0.92, 1.0)`): wycinek skalowany, potem **256×256**, sieć, mapa prawdopodobieństw **przeskalowana z powrotem** do wycinka i uśredniona,
   - lekkie **GaussianBlur** na mapie prawdopodobieństw przed progiem.  
   **Po co:** TTA stabilizuje predykcję; skala 0.92–1.0 daje niewielki zysk IoU na lokalnym zbiorze przy **braku** fałszywych alarmów na „good” (dobrane skryptem grid search).

5. **Próg U-Net** — `UNET_THRESHOLD = 0.30` (np. po strojeniu, nie domyślne 0.5).  
   **Po co:** lepszy kompromis precyzja/czułość na walidacji lokalnej.

6. **Post-processing maski sieci** (`_postprocess_unet_mask`) — morfologia close/open, **filtrowanie małych składowych spójnych** po powierzchni.  
   **Po co:** redukcja pojedynczych pikseli szumu.

7. **Łączenie** — OR klasyczki z maską ROI sieci; na końcu **AND z lekko dylatowaną maską ciała** — wycięcie przypadkowych pikseli poza szczoteczką.

**Evolucja w skrócie:** od prostszego pipeline’u (np. sam U-Net + prosty próg) → dodanie klasycznych gałęzi wewnętrznych/zewnętrznych → poprawy ścieżki wag → **TTA + wieloskalowość ROI + post-processing + dobór progu** oraz skrypty strojenia i wizualizacji (patrz sekcja 5).

---

## 5. Metryki, strojenie i wizualizacje (poza notebookami)

| Narzędzie | Rola |
|-----------|------|
| **`test_model.py`** | Import `predict` z wybranego katalogu zgłoszenia; **15 defective + 15 good** (lokalnie); IoU względem GT; podsumowanie średnich IoU; walidacja formatu wyjścia `{0,255}`, `uint8`, rozmiaru. |
| **`visualize_results.py`** | Dla defective: zapis **trójpanelowych** PNG (oryginał \| predykcja \| GT) do np. `visualization_results_defective/`; nagłówek z IoU, % wady, czas. |
| **`tune_multiscale_tta.py`** | Siatka **presetów** `(ROI_TTA_SCALES, ...)` przy **good IoU ≈ 1**; wybór sensownego kompromisu (lokalnie wybrano `(0.92, 1.0)`). |
| **`tune_threshold.py`** | Przegląd progów sigmoidy / binarnej maski dla U-Net przy zachowaniu ograniczeń na obrazy dobre. |
| **`tune_postprocess.py`** | Eksperymenty z rozmiarem dilatacji / morfologii na mapie prawdopodobieństw (wymaga spójności z aktualnym `model.py`). |

**Interpretacja:** średnia IoU lokalna (np. ~**0,77** na 30 obrazach w ostatniej konfiguracji) **nie musi** równać się wynikowi CodaBench na **innej** serii 42 obrazów — służy do **porównywania wersji** kodu względem siebie.

---

## 6. Zgłoszenie CodaBench — praktyka

- **ZIP:** `model.py`, `requirements.txt`, `weights.pth` (oraz ewentualne pliki wymagane przez organizatora) **w korzeniu archiwum**, nie w podfolderze `submission/`.
- Brak pobierania wag z internetu w czasie oceny.
- Spójność wersji bibliotek z `requirements.txt`; test lokalnie w kontenerze zbliżonym do produkcyjnego (Python 3.12) zmniejsza ryzyko.

---

## 7. Wnioski

- **Hybryda CV + U-Net** dobrze wykorzystuje duży kontrast tło–obiekt i geometrię rozchylenia; **wewnętrzne** wady są domeną sieci i ewentualnych uzupełnień (np. kanał jasności dla ciemniejszych obszarów).
- **TTA i wieloskalowość ROI** poprawiają IoU przy zachowaniu czystych mask na klasie „good”, o ile parametry są **dobrane na walidacji**, nie „na oko”.
- Trudne przypadki **jasna wada / jasne tło** wymagają przede wszystkim **danych i uczenia** (oversampling, augmentacje, ewentualnie kanały pomocnicze w treningu); agresywne reguły koloru–-tekstury w inferencji łatwo psują obrazy bez wad.
- Dokumentacja i powtarzalność: **dwa notebooki** (przygotowanie cropów + trening) + **jeden** pipeline inferencyjny `model.py` + skrypty testowe zamykają pętlę od eksperymentu do zgłoszenia.

---

*Raport: wersja robocza zsynchronizowana ze stanem repozytorium (inferencja `submission5`, skrypty testowe i strojące).*

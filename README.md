# Tracking Pipeline

Modulare Python-Pipeline fuer das Tracken von Objekten in LiDAR-Punktwolken, die spaetere Akkumulation von Track-Punktwolken und ein Open3D-basiertes Replay der Ergebnisse.

Das Repository ist aus dem Prototyp `cluster_track_lane_aggregate.py` heraus in eine saubere Schichtenarchitektur ueberfuehrt worden. Ziel ist, Tracking- und Akkumulationsalgorithmen austauschbar zu machen, ohne den zentralen Pipeline-Ablauf oder die Fachlogik neu schreiben zu muessen.

## Ziel des Repositories

Das Projekt loest drei Aufgaben in einem konsistenten Ablauf:

1. Punktwolken aus `a42.frame.Frame`-basierten `.pb`-Dateien lesen.
2. Bewegte bzw. relevante Objekte in einer Lane-Region clustern und ueber mehrere Frames tracken.
3. Pro Track Punktwolken ueber mehrere Beobachtungen akkumulieren und als Artefakte speichern.

Dazu kommt ein Replay-Modus, der denselben Runpfad erneut ausfuehrt und die Zustandsfolge in Open3D visualisiert.

## Aktueller Funktionsumfang

### Implementierte Eingangsformate
- `a42_pb`
  - length-delimited `Frame`-Messages
  - interner, im Repo vendorter Reader
  - keine externe `a42`-Installation zur Laufzeit erforderlich

### Implementierte Clusterer
- `dbscan`
  - Lane-Box-Crop auf Frame-Ebene
  - anschliessend `Open3D`-DBSCAN
  - Filter fuer minimale und maximale Clusterpunktzahl
- `euclidean_clustering`
  - euklidische Nachbarschaftsbaseline auf derselben Lane-Punktwolke
  - absichtlich einfacher Gegenpol zu DBSCAN fuer Benchmarking
- `ground_removed_dbscan`
  - RANSAC-Ebenensegmentierung fuer Bodenentfernung
  - danach DBSCAN auf den Nicht-Boden-Punkten
  - gibt Metriken zu entfernten Bodenpunkten zurueck
- `hdbscan`
  - optionale Benchmark-Variante fuer variable Punktdichten
  - nur verfuegbar, wenn das Extra `.[benchmark]` installiert wurde
- `range_image_connected_components`
  - echte Sensorraum-Clusterung auf Range-Image-Zellen
  - nutzt Scan-Kalibrierung und Sensorgrid-Indizes aus dem Reader
- `range_image_depth_jump`
  - Sensorraum-Clusterung mit expliziter Tiefensprung-Trennung
  - fuer harte Objektkanten im LiDAR-Bild gedacht
- `beam_neighbor_region_growing`
  - Region Growing nur entlang benachbarter Beams/Zellen
  - robuster gegen unphysikalische XYZ-Nachbarschaften ueber grosse Range-Spruenge

### Implementierte Tracker
- `euclidean_nn`
  - einfacher Nearest-Neighbor-Tracker auf Clusterzentren
  - keine Bewegungsmodellierung
  - guter, einfacher Baseline-Tracker
- `kalman_nn`
  - 3D-Kalman-Filter mit konstantem Geschwindigkeitsmodell
  - Nearest-Neighbor-Assignment auf vorhergesagte Positionen
  - Sticky-Gating ueber `missed`-Zaehler
- `hungarian_kalman`
  - gleicher Kalman-Unterbau wie `kalman_nn`
  - globale Zuordnung ueber Kostenmatrix und Hungarian-Assignment
  - neue Standardbaseline fuer fairere Assoziations-Benchmarks

### Implementierte Akkumulatoren
- `voxel_fusion`
  - Frame-Selektion pro Track
  - optional lokale oder Weltkoordinaten
  - voxelbasierte Fusion ueber mehrere Beobachtungen
  - statistischer Outlier-Filter und finaler Voxel-Downsample
- `registration_voxel_fusion`
  - wie `voxel_fusion`, aber mit Registrierungsstufe vor der Fusion
  - mit mehreren Registrierungsbackends
  - wenn Backend fehlt oder Registrierung verworfen wird, faellt die Pipeline kontrolliert zur unveraenderten Chunk-Verwendung zurueck
- `weighted_voxel_fusion`
  - voxelbasierte Fusion mit Gewichten pro Chunk
  - Gewichtung nach Punktzahl, Qualitaet oder uniform
- `occupancy_consensus_fusion`
  - behaelt nur Voxel, die ueber mehrere Beobachtungen konsistent sind
  - sinnvoll fuer robustere Offline-Fusion bei verrauschten Tracks

### Implementierte Registrierungsbackends
- `small_gicp`
  - optionales Extra wie im Prototyp
- `icp_point_to_plane`
  - stabile Open3D-Referenzvariante
- `generalized_icp`
  - Vergleichsbasis zu `small_gicp`
- `feature_global_then_local`
  - globales FPFH/RANSAC-Initialalignment, danach lokales ICP-Refinement

### Implementiertes Offline-Postprocessing
- `tracklet_stitching`
  - verbindet zeitlich nahe, raeumlich kompatible Track-Segmente
- `co_moving_track_merge`
  - fuehrt parallel laufende Cab-/Trailer- oder Front-/Bed-Teiltracks vor der Aggregation zusammen
- `trajectory_smoothing`
  - glaettet Track-Zentren ueber ein gleitendes Fenster
- `track_quality_scoring`
  - berechnet einen normierten Qualitaetsscore pro Track

### Persistente Run-Artefakte
Pro Run werden erzeugt:
- `config.snapshot.yaml`
- `summary.json`
- `tracks.jsonl`
- `aggregates/track_<id>.pcd`
- `aggregates/track_<id>.json`

### Benchmark-Artefakte
Pro Benchmark-Run werden erzeugt:
- `benchmarks/<timestamp>_<name>/results.csv`
- `benchmarks/<timestamp>_<name>/results.json`
- `benchmarks/<timestamp>_<name>/leaderboard.md`
- `benchmarks/<timestamp>_<name>/leaderboard_long_vehicle.md`
- `benchmarks/<timestamp>_<name>/performance_runs.jsonl`
- `benchmarks/<timestamp>_<name>/performance_leaderboard.md`
- `benchmarks/<timestamp>_<name>/performance_components.md`
- `benchmarks/<timestamp>_<name>/runs/...`

### Visualisierung
- Open3D-Einzelrun-Replay
- Anzeige von:
  - Lane-Punkten
  - Lane-Box
  - aktiven Track-Clustern
  - Track-Bounding-Boxes
  - Track-Zentren
  - optional aggregierter Punktwolke des aktuellen Track-IDsatzes

## Architektur

Die Architektur folgt einer klaren Trennung nach Verantwortlichkeiten.

### Presentation Layer
Die Presentation-Schicht ist in diesem Repo die CLI in [src/tracking_pipeline/cli.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/cli.py).

Verantwortung:
- CLI-Argumente parsen
- Konfiguration laden
- Use Cases starten
- Ergebnis kurz loggen

Sie enthaelt keine Cluster-, Tracking- oder Akkumulationslogik.

### Application Layer
Die Application-Schicht orchestriert den Ablauf, kennt aber keine konkreten Bibliotheksdetails.

Wichtige Dateien:
- [application/ports.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/ports.py)
- [application/factories.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/factories.py)
- [application/run_pipeline.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/run_pipeline.py)
- [application/replay_run.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/replay_run.py)

Verantwortung:
- Pipeline-Schritte in fester Reihenfolge ausfuehren
- Ports auf konkrete Implementierungen abbilden
- Run-Artefakte erzeugen lassen
- Replay aufbauen

### Domain Layer
Die Domain-Schicht enthaelt die fachlichen Typen und Regeln, ohne Abhaengigkeit auf Dateisystem, `Open3D` oder `betterproto`.

Wichtige Dateien:
- [domain/models.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/domain/models.py)
- [domain/value_objects.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/domain/value_objects.py)
- [domain/rules.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/domain/rules.py)
- [domain/events.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/domain/events.py)

Beispiele fuer Domain-Inhalte:
- `LaneBox`
- `FrameData`
- `Detection`
- `Track`
- `AggregateResult`
- Frame-Selektion fuer die Akkumulation
- Track-Exit-Regel
- Transform-Validierung

### Infrastructure Layer
Die Infrastructure-Schicht haengt die eigentlichen Bibliotheken und Algorithmen an die Ports an.

Bereiche:
- Reader
- Clusterer
- Tracker
- Akkumulatoren
- IO / Artefakt-Writer
- Visualisierung
- Logging

Hier leben `Open3D`, das vendorte `a42`-Protobufmodell und optionale Registrierungsbackends.

## Architekturregeln

Diese Regeln sind im Repo bewusst eingehalten:
- `domain` importiert keine IO- oder Visualisierungsbibliotheken.
- `application` orchestriert, implementiert aber keine Algorithmen.
- `infrastructure` implementiert Ports, startet aber keine Use Cases.
- Konfiguration ist zentralisiert und nicht als globale Konstanten ueber viele Dateien verteilt.
- Tracking- und Akkumulationsalgorithmen sind ueber kleine Registries in den Factories austauschbar.
- Registrierung ist ein optionales Detail des Akkumulators und keine Grundannahme der gesamten Pipeline.

## Datenfluss

Der aktuelle `run`-Datenfluss ist:

1. Config laden
2. Reader anhand von `input.format` erstellen
3. Lane-Box aus Config aufbauen
4. Frames aus `.pb` lesen
5. Punkte innerhalb der Lane-Box auswaehlen
6. konfigurierten Clusterer auf der Lane-Punktwolke ausfuehren
7. `ClusterResult` mit Detections und Cluster-Metriken erzeugen
8. Detections an den konfigurierten Tracker uebergeben
9. nach dem letzten Frame alle Tracks finalisieren
10. optionale Offline-Postprocessing-Schritte auf finale Tracks anwenden
11. pro Track den konfigurierten Akkumulator aufrufen
12. Ergebnisse als Run-Artefakte persistieren

Der `replay`-Datenfluss ist aehnlich, endet aber mit dem Open3D-Viewer statt mit einem rein persistierten Ergebnis.

Der `benchmark`-Datenfluss ist:
1. Benchmark-Manifest laden
2. kuratierte Preset-Matrix auf eine oder mehrere Sequenzen expandieren
3. pro Preset/Sequenz normalen `run` ausfuehren
4. Proxy-Metriken aus `summary.json` und `tracks.jsonl` aggregieren
5. `CSV`, `JSON` und `Markdown`-Leaderboard schreiben

## Projektstruktur

```text
tracking-pipeline/
├── pyproject.toml
├── README.md
├── configs/
├── runs/
├── src/
│   └── tracking_pipeline/
│       ├── cli.py
│       ├── config/
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       └── shared/
└── tests/
```

### Wichtige Verzeichnisse
- [configs](/Users/georg/workspace/tracking-pipeline/configs)
  - YAML-Presets fuer verschiedene Algorithmuskombinationen
- [src/tracking_pipeline/config](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/config)
  - Config-Dataclasses, Loader, Validierung
- [src/tracking_pipeline/domain](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/domain)
  - fachliche Modelle und Regeln
- [src/tracking_pipeline/application](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application)
  - Ports, Factories, Use Cases
- [src/tracking_pipeline/infrastructure/readers](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/readers)
  - PB-Reader und vendorte `a42`-Proto-Module
- [src/tracking_pipeline/infrastructure/clustering](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/clustering)
  - Clusterer
- [src/tracking_pipeline/infrastructure/tracking](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/tracking)
  - Tracking-Implementierungen
- [src/tracking_pipeline/infrastructure/aggregation](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/aggregation)
  - Akkumulatoren und Registrierungsbackend
- [src/tracking_pipeline/infrastructure/io](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/io)
  - Run-Artefakte schreiben
- [src/tracking_pipeline/infrastructure/visualization](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/visualization)
  - Open3D-Replay
- [tests](/Users/georg/workspace/tracking-pipeline/tests)
  - Unit- und Integrationstests

## Installation

### Basisinstallation
```bash
pip install -e .
```

### Mit optionalem Registrierungsbackend
```bash
pip install -e .[registration]
```

### Entwicklungsabhaengigkeiten
```bash
pip install -e .[dev]
```

### Mit optionalen Benchmark-Algorithmen
```bash
pip install -e .[benchmark]
```

## CLI

### Pipeline ausfuehren
```bash
tracking-pipeline run -c configs/kalman_voxel.yaml
```

Alternativ ohne installierten Console-Script-Eintrag:
```bash
PYTHONPATH=src python -m tracking_pipeline.cli run -c configs/kalman_voxel.yaml
```

### Replay starten
```bash
tracking-pipeline replay -c configs/kalman_voxel.yaml
```

### Benchmark starten
```bash
tracking-pipeline benchmark -c configs/benchmark_curated_real.yaml
```

### Long-Vehicle-Benchmark starten
```bash
tracking-pipeline benchmark -c configs/benchmark_long_vehicle_real.yaml
```

### Kalman-Long-Vehicle-Benchmark starten
```bash
tracking-pipeline benchmark -c configs/benchmark_long_vehicle_kalman_real.yaml
```

## Konfiguration

Die Konfiguration wird aus YAML geladen.

Wichtige Presets:
- [configs/base.yaml](/Users/georg/workspace/tracking-pipeline/configs/base.yaml)
- [configs/kalman_voxel.yaml](/Users/georg/workspace/tracking-pipeline/configs/kalman_voxel.yaml)
- [configs/kalman_registration.yaml](/Users/georg/workspace/tracking-pipeline/configs/kalman_registration.yaml)
- [configs/euclidean_voxel.yaml](/Users/georg/workspace/tracking-pipeline/configs/euclidean_voxel.yaml)
- [configs/hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/hungarian_weighted.yaml)
- [configs/ground_removed_icp.yaml](/Users/georg/workspace/tracking-pipeline/configs/ground_removed_icp.yaml)
- [configs/hdbscan_consensus.yaml](/Users/georg/workspace/tracking-pipeline/configs/hdbscan_consensus.yaml)
- [configs/kalman_generalized_icp.yaml](/Users/georg/workspace/tracking-pipeline/configs/kalman_generalized_icp.yaml)
- [configs/kalman_feature_global_then_local.yaml](/Users/georg/workspace/tracking-pipeline/configs/kalman_feature_global_then_local.yaml)
- [configs/range_image_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/range_image_hungarian_weighted.yaml)
- [configs/range_image_depth_jump_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/range_image_depth_jump_hungarian_weighted.yaml)
- [configs/beam_neighbor_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/beam_neighbor_hungarian_weighted.yaml)
- [configs/long_vehicle_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_hungarian_weighted.yaml)
- [configs/long_vehicle_range_image_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_range_image_hungarian_weighted.yaml)
- [configs/long_vehicle_beam_neighbor_hungarian_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_beam_neighbor_hungarian_weighted.yaml)
- [configs/long_vehicle_kalman_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_kalman_weighted.yaml)
- [configs/long_vehicle_range_image_kalman_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_range_image_kalman_weighted.yaml)
- [configs/long_vehicle_range_image_depth_jump_kalman_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_range_image_depth_jump_kalman_weighted.yaml)
- [configs/long_vehicle_beam_neighbor_kalman_weighted.yaml](/Users/georg/workspace/tracking-pipeline/configs/long_vehicle_beam_neighbor_kalman_weighted.yaml)
- [configs/benchmark_curated_real.yaml](/Users/georg/workspace/tracking-pipeline/configs/benchmark_curated_real.yaml)
- [configs/benchmark_long_vehicle_real.yaml](/Users/georg/workspace/tracking-pipeline/configs/benchmark_long_vehicle_real.yaml)
- [configs/benchmark_long_vehicle_kalman_real.yaml](/Users/georg/workspace/tracking-pipeline/configs/benchmark_long_vehicle_kalman_real.yaml)

Presets werden gegen `base.yaml` gemerged. Relative Pfade werden relativ zur jeweils geladenen YAML-Datei aufgeloest.

### Config-Sektionen

#### `input`
- `path`
  - Pfad zur `.pb`-Datei
- `format`
  - aktuell nur `a42_pb`

#### `preprocessing`
- `lane_box`
  - `[x_min, x_max, y_min, y_max, z_min, z_max]`
- `bootstrap_frames`
  - aktuell validiert und im Modell vorhanden, aber im produktiven Ablauf noch nicht aktiv verwendet

#### `clustering`
- `algorithm`
  - `dbscan`, `euclidean_clustering`, `ground_removed_dbscan`, `hdbscan`, `range_image_connected_components`, `range_image_depth_jump`, `beam_neighbor_region_growing`
- `eps`
  - Nachbarschafts-/Clusterradius fuer DBSCAN und Euclidean-Baseline
- `min_points`
  - DBSCAN-Mindestdichte
- `vehicle_min_points`
  - untere Punktgrenze fuer gueltige Cluster
- `vehicle_max_points`
  - obere Punktgrenze fuer gueltige Cluster
- `plane_distance_threshold`
  - RANSAC-Abstandsschwelle fuer Bodenentfernung
- `plane_ransac_n`
  - Anzahl Punkte pro RANSAC-Sample
- `plane_num_iterations`
  - Iterationen fuer die Ebenensuche
- `ground_normal_z_min`
  - Mindestanteil der z-Normalkomponente, damit die Ebene als Boden gilt
- `hdbscan_min_cluster_size`
  - minimale Clustergruesse fuer `hdbscan`
- `hdbscan_min_samples`
  - Stichprobenparameter fuer `hdbscan`
- `sensor_range_min`
  - minimale Distanz fuer Sensorraum-Clusterung
- `sensor_range_max`
  - maximale Distanz fuer Sensorraum-Clusterung
- `sensor_depth_jump_ratio`
  - relatives Tiefensprung-Kriterium fuer Range-Image-Clusterer
- `sensor_depth_jump_abs`
  - absolutes Tiefensprung-Kriterium fuer Range-Image-Clusterer
- `sensor_min_component_size`
  - minimale Groesse einer Rohkomponente im Sensorgrid
- `sensor_neighbor_rows`
  - vertikales Nachbarschaftsfenster im Sensorgrid
- `sensor_neighbor_cols`
  - horizontales Nachbarschaftsfenster im Sensorgrid
- `sensor_ground_row_ignore`
  - ignoriert die untersten Sensorzeilen vor der Clusterung

#### `tracking`
- `algorithm`
  - `euclidean_nn`, `kalman_nn` oder `hungarian_kalman`
- `max_dist`
  - maximales Matching-Gate
- `max_missed`
  - wie viele Frames ein Track ohne Match ueberlebt
- `min_track_hits`
  - minimale Trefferzahl, bevor der Track aggregiert werden darf
- `sticky_extra_dist_per_missed`
  - zusaetzliches Gating pro verpasstem Frame fuer `kalman_nn`
- `sticky_max_dist`
  - Obergrenze fuer Sticky-Gating
- `kf_init_var`
  - Initialvarianz des Kalman-Filters
- `kf_process_var`
  - Prozessrauschen
- `kf_meas_var`
  - Messrauschen
- `association_size_weight`
  - Zusatzgewicht fuer Cluster-Groesse in der Kostenmatrix

#### `aggregation`
- `algorithm`
  - `voxel_fusion`, `registration_voxel_fusion`, `weighted_voxel_fusion`, `occupancy_consensus_fusion`
- `frame_selection_method`
  - `auto`, `all_track_frames`, `line_touch_last_k`, `keyframe_motion`, `length_coverage`
- `use_all_frames`
  - alle Track-Frames aggregieren oder vorher selektieren
- `top_k_frames`
  - falls `use_all_frames=false`: wie viele der letzten relevanten Frames genutzt werden
- `keyframe_keep`
  - wie viele Bewegungs-Keyframes bei `keyframe_motion` behalten werden
- `frame_selection_line_axis`
  - Achse fuer die Touch-Line-Logik
- `frame_selection_line_ratio`
  - Position der Touch-Line relativ zur Lane-Box
- `frame_selection_touch_margin`
  - Toleranz fuer den Touch-Test
- `frame_downsample_voxel`
  - Vorab-Downsampling pro Chunk
- `shape_consistency_filter`
  - filtert Chunks mit stark abweichender Ausdehnung
- `shape_consistency_max_extent_ratio`
  - Toleranz fuer die Shape-Konsistenz
- `registration_backend`
  - `small_gicp`, `icp_point_to_plane`, `generalized_icp`, `feature_global_then_local`
- `registration_max_corr_dist`
  - Korrespondenzdistanzen fuer Registrierung
- `registration_max_iter`
  - Iterationslimit
- `registration_min_fitness`
  - Mindestfitness fuer Akzeptanz
- `registration_max_translation`
  - Transform-Grenze als Sicherheitscheck
- `global_registration_voxel`
  - Voxelgroesse fuer FPFH/RANSAC bei globalem Initialalignment
- `fusion_voxel_size`
  - Voxelgroesse fuer die eigentliche Fusion
- `fusion_min_observations`
  - minimale Beobachtungsanzahl pro Voxel
- `fusion_weight_mode`
  - `uniform`, `point_count`, `quality`
- `consensus_ratio`
  - benoetigte Beobachtungsquote fuer `occupancy_consensus_fusion`
- `min_track_quality_for_save`
  - Score-Schwelle fuer das Persistieren von Aggregat-Artefakten
- `long_vehicle_mode`
  - erzwingt long-vehicle-spezifische Selektion, Tail-Bridge und Quality-Gating
- `long_vehicle_length_threshold`
  - Schwellwert fuer die automatische Long-Vehicle-Erkennung entlang der Fahrtrichtungsachse
- `length_coverage_bins`
  - Diskretisierung fuer `length_coverage`, damit lange Fahrzeuge ueber ihre Ausdehnung abgedeckt werden
- `min_track_quality_for_save_long_vehicle`
  - eigene Persistenzschwelle fuer lange Fahrzeuge
- `tail_bridge_longitudinal_gap_max`
  - maximaler Lueckenabstand entlang der Fahrtrichtung fuer Tail-Bridge
- `tail_bridge_lateral_gap_max`
  - maximaler Querabstand fuer Tail-Bridge
- `tail_bridge_vertical_gap_max`
  - maximaler Hoehenabstand fuer Tail-Bridge
- `post_filter_stat_nb_neighbors`
  - Parameter des statistischen Outlier-Filters
- `post_filter_stat_std_ratio`
  - Parameter des statistischen Outlier-Filters
- `aggregate_voxel`
  - finales Downsampling der fusionierten Wolke

#### `postprocessing`
- `enable_tracklet_stitching`
  - verbindet kompatible Teiltracks nach dem Online-Tracking
- `stitching_max_gap`
  - maximaler zeitlicher Abstand fuer Stitching
- `stitching_max_center_dist`
  - maximaler Mittelpunktabstand fuer Stitching
- `enable_co_moving_track_merge`
  - fuehrt zeitlich ueberlappende, gemeinsam bewegte Teiltracks zusammen
- `parallel_merge_max_lateral_offset`
  - maximaler lateraler Offset fuer Cab-/Trailer-Merge
- `parallel_merge_max_longitudinal_gap`
  - maximaler longitudinaler Abstand fuer Cab-/Trailer-Merge
- `parallel_merge_min_overlap_frames`
  - minimale Anzahl ueberlappender Frames fuer Cab-/Trailer-Merge
- `parallel_merge_min_overlap_ratio`
  - minimale Ueberlappungsquote fuer Cab-/Trailer-Merge
- `enable_trajectory_smoothing`
  - glaettet finale Track-Zentren
- `smoothing_window`
  - Fensterbreite fuer die Glattungsoperation
- `enable_track_quality_scoring`
  - berechnet Qualitaetsscore und Detailmetriken pro Track

#### `output`
- `root_dir`
  - Basisordner fuer Runs
- `save_world`
  - `false`: lokale, zentrierte Trackpunkte
  - `true`: Weltkoordinaten aggregieren
- `require_track_exit`
  - nur Tracks speichern, die die Lane-Zone plausibel verlassen haben
- `track_exit_edge_margin`
  - Toleranz an der Boxkante

#### `visualization`
- `enabled`
  - aktuell im Ablauf noch nicht ausgewertet; `replay` startet den Viewer unabhaengig davon
- `max_points`
  - maximale Lane-/Aggregate-Punkte fuer den Viewer
- `max_cluster_points`
  - maximale Punkte pro Clusterdarstellung
- `max_assoc_dist`
  - aktuell im Viewer-/Replay-Pfad noch nicht aktiv verwendet

## Beispielkonfiguration

```yaml
input:
  paths:
    - ../tests/fixtures/sample_a42.pb
clustering:
  algorithm: euclidean_clustering
  eps: 0.25
tracking:
  algorithm: hungarian_kalman
aggregation:
  algorithm: weighted_voxel_fusion
  frame_selection_method: keyframe_motion
  fusion_weight_mode: point_count
postprocessing:
  enable_track_quality_scoring: true
```

Hinweis: Die aktuellen Presets verweisen auf `../data/3.pb` relativ zu `configs/`. Wenn diese Datei lokal nicht existiert, muss `input.paths` angepasst werden.

## Benchmark-Konfiguration

Eine Benchmark-Konfiguration enthaelt:
- `name`
  - Name des Benchmark-Runs
- `output_root`
  - Zielordner fuer Benchmark-Reports
- `warmup_runs`
  - Anzahl nicht gewerteter Aufwaermlaeufe pro Preset/Sequenz
- `measure_runs`
  - Anzahl gewerteter Messlaeufe pro Preset/Sequenz
- `sequences`
  - Liste von `.pb`-Sequenzen
- `presets`
  - Liste von Preset-YAMLs, die auf jede Sequenz angewendet werden

Die Standardmatrix in [benchmark_curated_real.yaml](/Users/georg/workspace/tracking-pipeline/configs/benchmark_curated_real.yaml) vergleicht bewusst nur stabile Baselines und neue Roadmap-Presets.

## Run-Artefakte im Detail

### `config.snapshot.yaml`
Die zur Laufzeit verwendete, aufgeloeste Konfiguration.

### `summary.json`
Enthaelt immer:
- `input_paths`
- `input_path` (abgeleitet, bei Mehrdatei-Runs = erster Eintrag aus `input_paths`)
- `tracker_algorithm`
- `accumulator_algorithm`
- `clusterer_algorithm`
- `frame_count`
- `finished_track_count`
- `saved_aggregates`
- `registration_attempts`
- `registration_accepted`
- `registration_rejected`
- `output_dir`
- `postprocessing_methods`
- `aggregate_status_counts`
- `track_quality_mean`
- `performance`

`performance` enthaelt:
- `total_wall_seconds`
- `total_cpu_seconds`
- `compute_wall_seconds`
- `compute_cpu_seconds`
- `io_wall_seconds`
- `peak_rss_mb`
- `stages`

`performance.stages` fuehrt fuer jeden Pipeline-Stage:
- `wall_seconds`
- `cpu_seconds`
- `call_count`

### `tracks.jsonl`
Eine Zeile pro Track mit:
- `track_id`
- `source_track_ids`
- `frame_ids`
- `hit_count`
- `age`
- `missed`
- `ended_by_missed`
- `quality_score`
- `quality_metrics`
- `selected_frame_ids`
- `aggregate_status`
- `aggregation_metrics`

### `aggregates/*.pcd`
Die finale aggregierte Punktwolke pro gespeichertem Track.

### `aggregates/*.json`
Metadaten zur gespeicherten Aggregatwolke:
- Track-ID
- Status
- selektierte Frames
- Metriken der Aggregation und ggf. Registrierung

## Tracking-Logik

### `euclidean_nn`
Arbeitsweise:
- nimmt die aktuellen Clusterzentren
- sucht pro bestehendem Track den naechsten gueltigen Cluster innerhalb `max_dist`
- erzeugt fuer nicht zugeordnete Detections neue Tracks
- beendet Tracks nach `max_missed`

Eigenschaften:
- einfach
- schnell
- keine Bewegungsvorhersage
- empfindlicher bei Aussetzern oder schnellen Bewegungen

### `kalman_nn`
Arbeitsweise:
- fuehrt pro Track einen Kalman-Zustand `[x, y, z, vx, vy, vz]`
- sagt die naechste Position voraus
- matched Detections gegen die Vorhersage
- erlaubt groessere Gates fuer Tracks mit `missed > 0`

Eigenschaften:
- robuster als `euclidean_nn`
- besser bei kurzzeitigen Aussetzern
- mehr Parameter und etwas mehr Komplexitaet

### `hungarian_kalman`
Arbeitsweise:
- verwendet dieselbe Kalman-Vorhersage wie `kalman_nn`
- baut eine Kostenmatrix aus Distanz und optionaler Groessenkomponente
- loest das Matching global mit dem Hungarian-Verfahren

Eigenschaften:
- robuster bei mehreren nahen Detections
- fairere Benchmark-Basis als greedy Matching
- etwas hoeherer Aufwand pro Frame

## Akkumulationslogik

### `voxel_fusion`
Ablauf:
1. Track anhand von `min_track_hits` und optionaler Exit-Regel pruefen
2. lokale oder Weltpunkte pro Track verwenden
3. Frames auswaehlen
4. Chunks vorab downsamplen
5. Punkte voxelweise fusionieren
6. statistisch filtern
7. final downsamplen

### `registration_voxel_fusion`
Ablauf:
1. identisch zu `voxel_fusion` bis zur Chunk-Vorbereitung
2. Chunks nacheinander gegen ein wachsendes Modell registrieren
3. nur bei ausreichender Fitness und gueltigem Transform akzeptieren
4. danach normal voxelbasiert fusionieren

Wichtig:
- Registrierung ist optional.
- Fehlt ein optionales Backend wie `small_gicp`, laeuft die Pipeline weiter und markiert die Registrierung als `unavailable`.
- Schlechte Transformationen fuehren nicht zum Abbruch des Runs.

### `weighted_voxel_fusion`
Ablauf:
1. identisch zu `voxel_fusion`
2. berechnet pro Chunk Gewichte
3. fusioniert Voxelmittlewerte gewichtet statt rein arithmetisch

Geeignet fuer:
- Benchmarking unterschiedlicher Gewichtsstrategien
- spaetere Kopplung an Registrierungs- oder Qualitaetsmetriken

Bei `fusion_weight_mode=quality`:
- geht `track_quality_score` direkt in die Chunk-Gewichte ein
- wird zusaetzlich eine Chunk-Konsistenz aus Punktzahl- und Formabweichung zum Track-Median berechnet
- ignoriert der Long-Vehicle-Pfad longitudinale Extent-Schwankung bewusst weitgehend, damit Trailer/Ladeflaechen nicht weggedrueckt werden
- kann `min_track_quality_for_save` oder `min_track_quality_for_save_long_vehicle` das Schreiben von Artefakten unterbinden
- kann Tail-Bridge finale Komponenten entlang der Fahrtrichtung verbinden

### `occupancy_consensus_fusion`
Ablauf:
1. identisch zu `voxel_fusion`
2. erhoeht die Mindestbeobachtung pro Voxel anhand `consensus_ratio`
3. behaelt nur wiederholt beobachtete Struktur

Geeignet fuer:
- robuste Offline-Fusion
- verrauschte oder inkonsistente Track-Chunks

## Offline-Postprocessing

### `tracklet_stitching`
- fuehrt finale Tracks nach dem Online-Pass zusammen
- nutzt zeitliche Luecke und Mittelpunktabstand als Merge-Kriterium

### `trajectory_smoothing`
- glaettet Track-Zentren mit gleitendem Mittel
- aktualisiert lokale Trackpunkte passend zu den neuen Zentren

### `co_moving_track_merge`
- fuehrt ueberlappende Tracks mit aehnlicher Bewegungsrichtung und stabilem Parallelversatz zusammen
- ist fuer Splits zwischen Zugfahrzeug und Anhaenger oder Kabine und Ladeflaeche gedacht

### `track_quality_scoring`
- berechnet pro Track einen normierten Score in `[0, 1]`
- basiert auf Kontinuitaet, Beobachtungslaenge, Punktzahlstabilitaet und achsenspezifischer Formstabilitaet
- markiert lange Fahrzeuge ueber `is_long_vehicle` und fuehrt `length_cv`, `width_cv`, `height_cv` und `cross_section_cv`

## Benchmarking

Der Benchmark-Runner erzeugt Proxy-Metriken pro Preset/Sequenz. Die Ausfuehrung erfolgt standardmaessig mit `warmup_runs: 1` und `measure_runs: 3`; nur Messlaeufe gehen in die aggregierten Reports ein.

Bestehende Proxy-Metriken:
- `frame_count`
- `finished_track_count`
- `saved_aggregates`
- `aggregate_save_rate`
- `registration_attempts`
- `registration_accept_rate`
- `track_quality_mean`
- `mean_selected_frames_per_saved_aggregate`
- `mean_points_per_saved_aggregate`
- `mean_longitudinal_extent_saved`
- `p90_longitudinal_extent_saved`
- `mean_component_count_saved`
- `mean_tail_bridge_count_saved`
- `long_vehicle_saved_count`
- `runtime_seconds`
- `aggregate_status_counts`

Neue Vergleichsfelder:
- `registration_backend`
- `frame_selection_method`
- `long_vehicle_mode`
- `representative_run_dir`

Neue Performance-Metriken:
- `total_wall_seconds_median`, `total_wall_seconds_min`, `total_wall_seconds_max`
- `total_cpu_seconds_median`, `total_cpu_seconds_min`, `total_cpu_seconds_max`
- `compute_wall_seconds_median`, `compute_wall_seconds_min`, `compute_wall_seconds_max`
- `compute_cpu_seconds_median`, `compute_cpu_seconds_min`, `compute_cpu_seconds_max`
- `io_wall_seconds_median`, `io_wall_seconds_min`, `io_wall_seconds_max`
- `peak_rss_mb_median`, `peak_rss_mb_min`, `peak_rss_mb_max`
- `wall_ms_per_frame_median`, `cpu_ms_per_frame_median`, `accumulate_wall_ms_per_track_median`
- `stage_<stage>_wall_seconds_median` und `stage_<stage>_cpu_seconds_median`

Neue Benchmark-Artefakte:
- `performance_runs.jsonl`
  - ein Rohdatensatz pro Messlauf
- `performance_leaderboard.md`
  - schnellste, rechenintensivste und speicherintensivste Presets
- `performance_components.md`
  - deskriptive und gematchte Vergleiche fuer `clusterer`, `tracker`, `accumulator` und `registration_backend`

Sortierung im Leaderboard:
1. `saved_aggregates` absteigend
2. `track_quality_mean` absteigend
3. `registration_accept_rate` absteigend
4. `runtime_seconds` aufsteigend

Sortierung im Long-Vehicle-Leaderboard:
1. `mean_longitudinal_extent_saved` absteigend
2. `mean_component_count_saved` aufsteigend
3. `saved_aggregates` absteigend
4. `runtime_seconds` aufsteigend

## Visualisierung und Bedienung

Der Replay-Viewer ist in [open3d_replay_viewer.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/visualization/open3d_replay_viewer.py) implementiert.

Tastatursteuerung:
- `Space`
  - Play/Pause
- `N`
  - naechster Frame
- `B`
  - vorheriger Frame
- `A`
  - aggregierte Punktwolke ein-/ausblenden

Darstellung:
- Lane-Punkte: grau
- Track-Cluster: farbig
- Bounding-Boxes: farbig
- Track-Marker: farbige Kugeln
- Aggregatwolken: rot

## Tests

Tests ausfuehren:
```bash
pytest
```

Aktuell abgedeckt:
- Domain-Regeln
- Reader und Tracker
- Akkumulatoren
- Clusterer und Offline-Postprocessing
- Orchestrierung mit Fakes
- Integrationspfade fuer Algorithmuswechsel
- Benchmark-Runner und Report-Artefakte

Wichtige Testdateien:
- [tests/unit/domain/test_rules.py](/Users/georg/workspace/tracking-pipeline/tests/unit/domain/test_rules.py)
- [tests/unit/infrastructure/test_reader_and_trackers.py](/Users/georg/workspace/tracking-pipeline/tests/unit/infrastructure/test_reader_and_trackers.py)
- [tests/unit/infrastructure/test_accumulators.py](/Users/georg/workspace/tracking-pipeline/tests/unit/infrastructure/test_accumulators.py)
- [tests/unit/infrastructure/test_clusterers_and_postprocessing.py](/Users/georg/workspace/tracking-pipeline/tests/unit/infrastructure/test_clusterers_and_postprocessing.py)
- [tests/unit/application/test_run_pipeline_orchestration.py](/Users/georg/workspace/tracking-pipeline/tests/unit/application/test_run_pipeline_orchestration.py)
- [tests/integration/test_run_pipeline.py](/Users/georg/workspace/tracking-pipeline/tests/integration/test_run_pipeline.py)
- [tests/integration/test_algorithm_switching.py](/Users/georg/workspace/tracking-pipeline/tests/integration/test_algorithm_switching.py)
- [tests/integration/test_benchmark_runner.py](/Users/georg/workspace/tracking-pipeline/tests/integration/test_benchmark_runner.py)

## Erweiterung des Systems

### Neuen Clusterer hinzufuegen
1. Neue Datei in [infrastructure/clustering](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/clustering) anlegen.
2. `Clusterer`-Port implementieren und `ClusterResult` zurueckgeben.
3. Config-Validierung und Factory erweitern.
4. Unit- und Integrations-Tests ergaenzen.

### Neuen Tracker hinzufuegen
1. Neue Datei in [infrastructure/tracking](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/tracking) anlegen.
2. Port-Vertrag aus [application/ports.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/ports.py) implementieren.
3. Namen in [config/validation.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/config/validation.py) freischalten.
4. Factory in [application/factories.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/factories.py) erweitern.
5. Tests ergaenzen.

### Neuen Akkumulator hinzufuegen
1. Neue Datei in [infrastructure/aggregation](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/aggregation) anlegen.
2. `Accumulator`-Port implementieren.
3. Falls noetig Registrierungsbackend kapseln.
4. Config-Validierung und Factory erweitern.
5. Integrations- und Unit-Tests ergaenzen.

### Neuen Track-Postprocessor hinzufuegen
1. Neue Datei in [infrastructure/postprocessing](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/postprocessing) anlegen.
2. `TrackPostprocessor`-Port implementieren.
3. In [application/factories.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/factories.py) verdrahten.
4. Metriken in `tracks.jsonl` oder `summary.json` erweitern, wenn noetig.

### Neuen Reader hinzufuegen
1. Reader in [infrastructure/readers](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/readers) implementieren.
2. `FrameData` als gemeinsames Format beibehalten.
3. Neues `input.format` in Validierung und Factory aufnehmen.

## Bekannte Grenzen des aktuellen Stands

Diese Punkte sind bewusst wichtig fuer die korrekte Erwartung an das Repo:
- Nur ein Clusterer ist implementiert: `dbscan`.
- `preprocessing.bootstrap_frames` ist bereits modelliert, wird aber im aktuellen Laufpfad noch nicht verwendet.
- `visualization.enabled` wird aktuell nicht zur Steuerung des Replay-Aufrufs benutzt.
- `visualization.max_assoc_dist` ist aktuell noch ohne Wirkung.
- `replay` laedt noch keinen gespeicherten Run, sondern berechnet den Replay-Zustand erneut aus Input und Config.
- `registration_backend` ist derzeit praktisch auf `small_gicp` beschraenkt.
- Je nach Daten und Schwellwerten kann ein Run keine gespeicherten Aggregate erzeugen, obwohl Reader, Tracking und Persistenz technisch korrekt funktionieren.

## Relevante Einstiegspunkte im Code

Wenn du schnell in den Code einsteigen willst, sind diese Dateien die wichtigsten:
- CLI: [cli.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/cli.py)
- Config-Loader: [loader.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/config/loader.py)
- Pipeline-Orchestrierung: [run_pipeline.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/run_pipeline.py)
- Replay-Orchestrierung: [replay_run.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/application/replay_run.py)
- Clusterer: [dbscan_clusterer.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/clustering/dbscan_clusterer.py)
- Tracker: [euclidean_nn.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/tracking/euclidean_nn.py), [kalman_nn.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/tracking/kalman_nn.py)
- Akkumulatoren: [voxel_fusion.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/aggregation/voxel_fusion.py), [registration_voxel_fusion.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/aggregation/registration_voxel_fusion.py)
- Artefakt-Writer: [artifact_writer.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/io/artifact_writer.py)
- Viewer: [open3d_replay_viewer.py](/Users/georg/workspace/tracking-pipeline/src/tracking_pipeline/infrastructure/visualization/open3d_replay_viewer.py)

## Status

Der aktuelle Stand ist lauffaehig und getestet:
- modulare Architektur implementiert
- CLI vorhanden
- Tests vorhanden
- Algorithmuswechsel fuer Tracking und Akkumulation vorhanden
- Open3D-Replay vorhanden
- Run-Artefakte vorhanden

Fuer produktive Nutzung auf echten Daten ist der naechste Schritt typischerweise das Tuning der Schwellwerte und die Validierung auf laengeren `.pb`-Sequenzen.

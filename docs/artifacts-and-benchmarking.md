# Artefakte und Benchmarking

Diese Seite beschreibt, welche Dateien `run`, `replay` und `benchmark` erzeugen und welche Metriken darin besonders relevant sind.

## Run-Artefakte

Ein typischer Run unter `runs/<timestamp>_<tracker>_<accumulator>/` sieht so aus:

```text
runs/<run_name>/
├── config.snapshot.yaml
├── class_stats.json
├── summary.json
├── tracker_debug.jsonl
├── track_outcomes.jsonl
├── tracks.jsonl
├── aggregates/
│   ├── track_0001.pcd
│   └── track_0001.json
└── object_list/
    └── manifest.jsonl
```

## `config.snapshot.yaml`

- eingefrorene, bereits aufgeloeste Run-Konfiguration
- hilfreich fuer Reproduzierbarkeit und spaetere Vergleiche

## `summary.json`

`summary.json` fasst den gesamten Run zusammen.

Wichtige Felder:

| Feld | Bedeutung |
| --- | --- |
| `input_path` / `input_paths` | verarbeitete Eingabequellen |
| `tracker_algorithm` | effektiver Tracker |
| `clusterer_algorithm` | effektiver Clusterer |
| `accumulator_algorithm` | effektiver Akkumulator |
| `frame_count` | verarbeitete Frames |
| `finished_track_count` | finalisierte Tracks |
| `saved_aggregates` | gespeicherte Aggregate |
| `registration_attempts` | Anzahl Registrierungsversuche |
| `registration_accepted` | akzeptierte Registrierungen |
| `registration_rejected` | abgelehnte Registrierungen |
| `aggregate_status_counts` | Statusverteilung ueber alle Tracks |
| `track_quality_mean` | mittlere Track-Qualitaet |
| `predicted_class_counts` | Klassenverteilung der final gespeicherten Net-Vorhersagen; bei aktiver Klassen-Normalisierung im kanonischen Zielraum |
| `gt_class_counts` | Klassenverteilung der vorhandenen GT-Objektlabels im Run; bei aktiver Klassen-Normalisierung im kanonischen Zielraum |
| `matched_gt_class_counts` | Klassenverteilung nur ueber gematchte Saved-Tracks; bei aktiver Klassen-Normalisierung im kanonischen Zielraum |
| `class_comparison_count` | Anzahl gematchter Saved-Tracks mit gesetzter Prediction- und GT-Klasse, die direkt verglichen wurden |
| `class_match_count` | Anzahl dieser Vergleiche, bei denen Prediction und GT dieselbe Klasse haben |
| `class_mismatch_count` | Anzahl dieser Vergleiche, bei denen Prediction und GT unterschiedliche Klassen haben |
| `class_count_rows` | tabellarische Zeilen fuer `Class | Net | GT matches` inklusive `TOTAL` |
| `output_dir` | Zielverzeichnis des Runs |

## `class_stats.json`

`class_stats.json` enthaelt dieselbe kompakte Klassenstatistik wie `summary.json`, aber separat als leicht auswertbare Datei:

- `predicted_class_counts`: Anzahl gespeicherter Aggregate pro vorhergesagter Klasse
- `gt_class_counts`: Anzahl vorhandener GT-Objektlabels pro GT-Klasse
- `matched_gt_class_counts`: Anzahl gematchter Saved-Tracks pro GT-Klasse
- `class_comparison_count`: Anzahl direkter Klassenvergleiche zwischen Prediction und GT
- `class_match_count`: Anzahl gleicher Klassen unter diesen Vergleichen
- `class_mismatch_count`: Anzahl ungleicher Klassen unter diesen Vergleichen
- `class_count_rows`: tabellarische Zeilen fuer CLI und Datei-Auswertung

`predicted_class_counts` zaehlt nur finale `saved`-Aggregate mit gesetzter `predicted_class_name`. `gt_class_counts` basiert auf den exportierten, nicht-leeren GT-Objektlabels des Runs. `matched_gt_class_counts` und die GT-Spalte in `class_count_rows` zaehlen dagegen nur gematchte Saved-Tracks. `class_comparison_count`, `class_match_count` und `class_mismatch_count` betrachten ebenfalls nur gematchte Saved-Tracks, bei denen sowohl Prediction als auch GT-Klasse vorhanden sind. Wenn `class_normalization.enabled=true`, werden Prediction und GT vorher auf dieselbe Zieltaxonomie normalisiert.

### `summary.json.performance`

Seit der Performance-Erweiterung schreibt `run` ein verschachteltes `performance`-Objekt:

| Feld | Bedeutung |
| --- | --- |
| `total_wall_seconds` | gesamte Laufzeit als Wall-Clock |
| `total_cpu_seconds` | gesamte CPU-Zeit des Prozesses |
| `compute_wall_seconds` | Wall-Zeit der Rechenstages |
| `compute_cpu_seconds` | CPU-Zeit der Rechenstages |
| `io_wall_seconds` | Wall-Zeit von Lesen und Schreiben |
| `total_hz` | End-to-End-Durchsatz in Frames pro Sekunde (`frame_count / total_wall_seconds`) |
| `compute_hz` | Compute-Durchsatz in Frames pro Sekunde (`frame_count / compute_wall_seconds`) |
| `peak_rss_mb` | maximaler Resident Set Size in MB |
| `stages` | Timing pro Stage, z. B. `cluster_frames`, `tracker_steps`, `accumulate_tracks`, `classify_aggregates` |
| `aggregation_components` | aufsummierte Teilzeiten der Aggregation fuer `registration`, `fusion_core`, `post_filter`, `tail_bridge`, `confidence_cap`, `symmetry_completion`, `fusion_post`, `fusion_total` |

`compute_cpu_seconds` misst nur die eigentlichen Compute-Stages, nicht den gesamten Run.
`compute_hz` kann deshalb hoeher sein als `total_hz`, weil Lesen und Schreiben nicht enthalten sind.

`accumulate_tracks` bleibt die uebergeordnete Stage-Zeit. Sie kann groesser sein als `registration + fusion_total`, weil Selektion, Vorbereitung und andere Logik davor bzw. dazwischen bewusst nicht in den Teilkomponenten stecken.

## `tracks.jsonl`

Eine Zeile pro Track.

Top-Level-Felder:

| Feld | Bedeutung |
| --- | --- |
| `track_id` | Track-ID |
| `source_track_ids` | urspruengliche IDs nach Stitching/Merge |
| `frame_ids` | alle Frames des Tracks |
| `hit_count` | Anzahl Beobachtungen |
| `age` | Lebensdauer in Frames |
| `missed` | verpasste Frames vor Finalisierung |
| `ended_by_missed` | ob der Track durch Misses beendet wurde |
| `quality_score` | finaler Track-Qualitaetswert |
| `quality_metrics` | Detailmetriken zum Track |
| `tracker_debug_summary` | kompakte Spawn-/Match-/Miss-Statistik des Tracks |
| `decision_stage` | normalisierte Stufe der finalen Save-/Skip-Entscheidung |
| `decision_reason_code` | stabiler Failure-/Save-Code wie `track_exit`, `min_hits`, `saved` |
| `decision_summary` | kurze menschenlesbare Zusammenfassung fuer Replay/HUD |
| `last_frame_id` | letzter Frame des Tracks |
| `last_center` | letzter Track-Zentrumspunkt |
| `selected_frame_ids` | Chunks/Frames, die in die Aggregation eingegangen sind |
| `aggregate_status` | `saved` oder Skip-/Empty-Status |
| `predicted_class_id` / `predicted_class_name` / `predicted_class_score` | optionale PointNeXt-Top-1-Vorhersage fuer das finale Aggregat; `predicted_class_name` kann bei aktiver Klassen-Normalisierung bereits kanonisch sein |
| `classification_backend` / `classification_point_source` / `classification_input_point_count` | Debug-Kontext zur Klassifikation |
| `gt_obj_class` / `gt_obj_class_score` | optionale GT-Klasse des gematchten Objektlabels fuer gespeicherte Tracks; bei aktiver Klassen-Normalisierung kanonisch |
| `aggregation_metrics` | Detailmetriken der Aggregation |

## `tracker_debug.jsonl`

Eine Zeile pro Frame mit den Tracker-Entscheidungen des Replays/Run-Loops.

Wichtige Felder:

| Feld | Bedeutung |
| --- | --- |
| `frame_index` | verarbeiteter Frame |
| `cluster_metrics` | Clusterer-Metriken desselben Frames |
| `tracker_metrics` | kompakte Frame-Summary wie `matched_count`, `spawned_count`, `spawn_suppressed_count` |
| `tracker_debug.track_states` | Vorhersagen und Status pro Track (`matched`, `missed`, `spawned`) |
| `tracker_debug.detection_states` | Status pro Detection (`matched`, `spawned`, `unmatched`, `spawn_suppressed`) |

Das ist die beste Datei, um spaeter ohne Viewer nachzuvollziehen, warum ein zweiter Track entstanden ist.

## `track_outcomes.jsonl`

Eine Zeile pro finalisiertem Track mit kompakter Save-/Failure-Diagnose.

Wichtige Felder:

| Feld | Bedeutung |
| --- | --- |
| `track_id` | Track-ID |
| `status` | finaler `AggregateResult.status` |
| `decision_stage` | normalisierte Gate-Stufe, z. B. `tracking_gate`, `save_gate`, `saved` |
| `decision_reason_code` | stabiler Failure-/Save-Code |
| `decision_summary` | kurze Zusammenfassung fuer Replay und schnelle Diagnose |
| `last_frame_id` | letzter beobachteter Frame des Tracks |
| `last_playback_index` | Playback-Index fuer Replay-Overlays |
| `last_center` | letzter bekannter Mittelpunkt des Tracks |
| `hit_count` / `age` / `missed` | kompakte Lifecycle-Daten des Tracks |
| `ended_by_missed` | ob der Track durch Misses finalisiert wurde |
| `quality_score` | finaler Track-Qualitaetswert |
| `selected_frame_ids` | Frames, die fuer die Aggregation selektiert wurden |
| `predicted_class_id` / `predicted_class_name` / `predicted_class_score` | optionale PointNeXt-Vorhersage fuer Replay-Labels; `predicted_class_name` kann bei aktiver Klassen-Normalisierung kanonisch sein |
| `classification_backend` / `classification_point_source` / `classification_input_point_count` | Debug-Kontext zur Klassifikation |
| `gt_obj_class` / `gt_obj_class_score` | optionale GT-Klasse des gematchten Objektlabels fuer Replay-Labels; bei aktiver Klassen-Normalisierung kanonisch |
| `tracker_debug_summary` | Spawn-/Match-/Miss-Summen des Track-Lifecycles |

Das ist die schnellste Datei, um zu sehen, warum ein Track nicht gespeichert wurde, ohne erst alle Frame-Logs zu lesen.

### Wichtige `aggregation_metrics`

#### Geometrie und Save-Status

| Feld | Bedeutung |
| --- | --- |
| `decision_stage` | normalisierte Entscheidungsstufe des finalen Status |
| `decision_reason_code` | stabiler Reason-Code fuer Replay/Reports |
| `decision_summary` | kurze menschenlesbare Save-/Skip-Begruendung |
| `hit_count` / `min_track_hits` | Ist-/Soll-Werte fuer das Min-Hits-Gate |
| `track_exited` / `track_exit_edge_margin` / `distance_to_exit_line` | Ist-/Soll-Werte fuer das Exit-Gate |
| `track_exit_line_axis` / `track_exit_line_side` / `track_exit_line_value` | geometrische Definition der Exit-Linie |
| `track_center_line_coordinate` / `track_passed_exit_line` | letzter Centerwert auf der Exit-Achse und Crossing-Status |
| `selected_frame_count` / `prepared_chunk_count` | Selektion und Vorbereitung vor der Fusion |
| `point_count_after_fusion` | Punktzahl nach Voxel-Fusion |
| `point_count_after_stat_filter` | Punktzahl nach Statistical Outlier Removal |
| `point_count_after_downsample` | finale gespeicherte Punktzahl |
| `longitudinal_extent` | Ausdehnung auf der Lane-Laengsachse |
| `component_count_post_fusion` | Anzahl Komponenten nach Fusion |
| `tail_bridge_count` | erzeugte Bridge-Punkte fuer Long Vehicles |
| `mode` | `local` oder `world` |
| `predicted_class_id` / `predicted_class_name` / `predicted_class_score` | optionale PointNeXt-Top-1-Vorhersage im Aggregat; `predicted_class_name` kann bei aktiver Klassen-Normalisierung kanonisch sein |
| `classification_backend` / `classification_point_source` / `classification_input_point_count` | Backend, verwendete Punktquelle (`result_points` oder `candidate_points_world`) und Rohpunktzahl fuer die Inferenz |
| `gt_obj_class` / `gt_obj_class_score` | optionale GT-Klasse des gematchten Objektlabels; nur fuer gematchte gespeicherte Tracks vorhanden und bei aktiver Klassen-Normalisierung kanonisch |
| `registration_wall_seconds` / `registration_cpu_seconds` | Zeit fuer `_prepare_for_fusion(...)` bei registrierungsbasierten Akkumulatoren |
| `fusion_core_wall_seconds` / `fusion_core_cpu_seconds` | reine Zeit der Voxel-Akkumulation (`_fuse_chunks`) |
| `post_filter_wall_seconds` / `post_filter_cpu_seconds` | Zeit fuer `_post_filter(...)` inklusive optionalem Statistical Outlier Removal und `aggregate_voxel` |
| `tail_bridge_wall_seconds` / `tail_bridge_cpu_seconds` | Zeit fuer `_apply_tail_bridge(...)` inklusive Komponentenbildung und optionalem Bridge-Re-Filter |
| `confidence_cap_wall_seconds` / `confidence_cap_cpu_seconds` | Zeit fuer `_apply_confidence_point_cap(...)` |
| `symmetry_completion_wall_seconds` / `symmetry_completion_cpu_seconds` | Zeit fuer `_apply_symmetry_completion(...)` |
| `fusion_post_wall_seconds` / `fusion_post_cpu_seconds` | Nachlauf ab erstem Post-Filter bis zur finalen Punktwolke vor den Save-/Skip-Gates |
| `fusion_total_wall_seconds` / `fusion_total_cpu_seconds` | Summe aus `fusion_core` und `fusion_post` |

`fusion_post` bleibt der aeussere Gesamt-Bucket ueber den kompletten Nachlauf. Er kann groesser sein als `post_filter + tail_bridge + confidence_cap + symmetry_completion`, weil Dimensionsmetriken, Candidate-Bildung und Save-/Skip-Gates bewusst nur im Outer-Bucket stecken.

#### Fahrzeugdimensionen

| Feld | Bedeutung |
| --- | --- |
| `vehicle_length` | finale Aggregate-Laenge entlang der Longitudinalachse |
| `vehicle_width` | finale Breite entlang der lateralen Achse |
| `vehicle_height` | finale Hoehe entlang der vertikalen Achse |
| `vehicle_length_axis` | Achsenname der Laenge |
| `vehicle_width_axis` | Achsenname der Breite |
| `vehicle_height_axis` | Achsenname der Hoehe |
| `extent_x` / `extent_y` / `extent_z` | rohe xyz-Ausdehnungen der finalen Punktwolke |

#### Registrierung

| Feld | Bedeutung |
| --- | --- |
| `registration_backend` | effektives Registrierungsbackend |
| `registration_pairs` | registrierte Paare |
| `registration_accepted` | akzeptierte Paare |
| `registration_rejected` | verworfene Paare |
| `registration_input_chunk_count` | Chunks vor Registrierung |
| `registration_output_chunk_count` | effektiv weiterverwendete Chunks nach eventuellem Underfill-Fallback |
| `registration_dropped_count` | effektiv verworfene Chunks nach eventuellem Underfill-Fallback |
| `registration_keep_indices` | Indizes der effektiv behaltenen Chunks |
| `registration_chunk_weights` | finale Chunk-Gewichte fuer die Fusion |
| `registration_fallback_applied` | zeigt, ob auf unregistrierte selektierte Chunks zurueckgefallen wurde |
| `registration_fallback_min_kept_chunks` | Schwellwert fuer den optionalen Underfill-Fallback |
| `registration_attempt_output_chunk_count` | rohe Anzahl Registration-Chunks vor eventuellem Fallback |
| `registration_attempt_dropped_count` | rohe Anzahl verworfener Registration-Chunks vor eventuellem Fallback |
| `registration_attempt_keep_indices` | rohe behaltene Registration-Indizes vor eventuellem Fallback |
| `registration_attempt_chunk_weights` | rohe Registration-Gewichte vor eventuellem Fallback |

#### Symmetry Completion

| Feld | Bedeutung |
| --- | --- |
| `symmetry_completion_enabled` | Feature-Toggle war aktiv |
| `symmetry_completion_applied` | Completion wurde wirklich angewendet |
| `symmetry_completion_plane_coordinate` | verwendete Symmetrieebene |
| `symmetry_completion_source_side` | Seite, die als Quelle gespiegelt wurde |
| `point_count_before_symmetry_completion` | Punktzahl vor Completion |
| `symmetry_completion_generated_points` | neu erzeugte Spiegelpunkte |
| `point_count_after_symmetry_completion` | Punktzahl nach Completion |
| `motion_deskew_enabled` | Deskew-Toggle war aktiv |
| `motion_deskew_applied` | Deskew wurde auf mindestens einen Chunk angewendet |
| `motion_deskew_skipped_reason` | Debug-Grund fuer Skip oder `applied` |
| `motion_deskew_corrected_chunk_count` | Anzahl wirklich korrigierter Chunks |
| `motion_deskew_mean_speed_mps` | mittlere projizierte Lane-Geschwindigkeit der korrigierten Chunks |
| `motion_deskew_mean_abs_shift_m` | mittlere absolute Punktverschiebung durch den Deskew |
| `motion_deskew_mean_time_span_ms` | mittlere intra-chunk Zeitspanne der korrigierten Punkte |

## `aggregates/*.pcd`

- finale aggregierte Punktwolke pro gespeichertem Track
- optional mit PCD-Feld `reflectivity`, wenn `output.save_aggregate_intensity=true`
- Reflectivity wird beim Einlesen als `signal * r^2` berechnet; im aktuellen Datensatz stammt das Signal aus `pointcloud.reflectivity`

## `aggregates/*.json`

Zu jeder gespeicherten Punktwolke existiert eine JSON-Datei mit:

- `track_id`
- `status`
- `selected_frame_ids`
- `metrics` inklusive optionaler Klassifikations- und GT-Felder wie `predicted_class_name` oder `gt_obj_class`

Diese Datei ist die leichteste Stelle, um Aggregate-Metriken ohne PCD-Parser zu lesen.

## `object_list/`

Falls Objektlabels im Input vorhanden sind, wird zusaetzlich `object_list/` geschrieben:

- `object_<id>.pcd`
- `manifest.jsonl`

Das ist bewusst getrennt von den Track-Aggregaten und bleibt XYZ-only.

## `gt_matching/`

Falls GT-Objektlabels vorhanden sind, wird zusaetzlich `gt_matching/` geschrieben:

- `matches.jsonl`
- `unmatched_saved_tracks.jsonl`
- `unmatched_gt_objects.jsonl`
- `summary.json`

`matches.jsonl` und `unmatched_gt_objects.jsonl` enthalten neben IDs und Zeitstempeln auch `gt_obj_class` und `gt_obj_class_score`, falls das Objektlabel eine Klasse traegt.

## Benchmark-Artefakte

Ein Benchmark unter `benchmarks/<timestamp>_<name>/` erzeugt:

```text
benchmarks/<benchmark_name>/
├── results.csv
├── results.json
├── leaderboard.md
├── leaderboard_long_vehicle.md
├── performance_runs.jsonl
├── performance_leaderboard.md
├── performance_components.md
├── resolved_configs/
└── runs/
```

## Benchmark-Dateien

### `results.csv` und `results.json`

Aggregierte Vergleichsdaten pro Preset und Sequenz.

Wichtige Spalten:

| Feld | Bedeutung |
| --- | --- |
| `sequence_name` | Sequenzname |
| `preset_name` | Presetname |
| `tracker_algorithm` | Tracker des Presets |
| `clusterer_algorithm` | Clusterer des Presets |
| `accumulator_algorithm` | Akkumulator des Presets |
| `registration_backend` | Registration-Backend des Presets |
| `frame_selection_method` | Frame-Selektionsstrategie |
| `long_vehicle_mode` | Long-Vehicle-Toggle |
| `saved_aggregates` | gespeicherte Aggregate |
| `aggregate_save_rate` | Anteil gespeicherter Aggregate |
| `mean_points_per_saved_aggregate` | mittlere finale Punktzahl |
| `mean_longitudinal_extent_saved` | mittlere Laengsausdehnung gespeicherter Aggregate |
| `mean_component_count_saved` | mittlere Komponentenanzahl |
| `runtime_seconds` | Median der gesamten Wall-Zeit |

### Performance-Felder in `results.*`

Performancewerte werden als Median/Min/Max ueber die `measure_runs` geschrieben. Standardmaessig ist das jetzt ein einzelner Messlauf; wiederholte Laufzeitmessungen sind opt-in ueber `measure_runs > 1`.

| Feldtyp | Beispiel |
| --- | --- |
| Gesamtzeiten | `total_wall_seconds_median`, `compute_cpu_seconds_max` |
| Durchsatz | `total_hz_median`, `compute_hz_max` |
| normalisierte Zeiten | `wall_ms_per_frame_median`, `accumulate_wall_ms_per_track_median` |
| Stage-Zeiten | `stage_cluster_frames_wall_seconds_median` |

Warmups gehen **nicht** in diese Aggregationen ein.

### `performance_runs.jsonl`

- Rohdatensatz mit einer Zeile pro Messlauf
- nuetzlich fuer Ausreisseranalyse und Debugging einzelner Runs

### `performance_leaderboard.md`

Gruppiert Presets nach:

- schnellsten Gesamtlaeufen
- rechenintensivsten Presets
- speicherintensivsten Presets
- inklusive `Total Hz` und `Compute Hz` als zusaetzliche Durchsatzspalten

### `performance_components.md`

Komponentenvergleich auf zwei Ebenen:

- deskriptive Gruppierung nach Komponentenfamilien
- gematchte Vergleiche bei gleicher Restkonfiguration
- inklusive `Total Hz`, `Compute Hz`, `ICP/Registration`, `Fusion Total`, `Post-Filter`, `Tail-Bridge`, `Conf-Cap` und `Symmetry` fuer direkte Laufzeitvergleiche

Das ist die beste Stelle fuer Fragen wie:

- Welcher Tracker ist am schnellsten?
- Welches Registration-Backend kostet am meisten CPU?
- Wie viel Zeit geht in ICP/Registration gegenueber der eigentlichen Fusion?
- Welcher Teil von `fusion_post` dominiert: `post_filter`, `tail_bridge`, `confidence_cap` oder `symmetry_completion`?
- Welche Presets haben den groessten Peak-Speicher?

## Praktische Auswertung

Fuer einen schnellen Vergleich:

1. `leaderboard.md` fuer qualitative Reihenfolge lesen.
2. `performance_leaderboard.md` fuer Laufzeit und Speicher lesen.
3. `performance_components.md` verwenden, wenn nur ein Teilalgorithmus verglichen werden soll.
4. `tracks.jsonl` oeffnen, wenn ein einzelner Track auffaellig ist.

## Typische Fehlinterpretationen vermeiden

- `summary.json` beschreibt **einen** Run, nicht den Median mehrerer Runs.
- `results.csv` beschreibt Benchmark-Aggregate ueber mehrere Messlaeufe.
- `compute_cpu_seconds` ist nicht gleich `total_cpu_seconds`: es umfasst nur die Compute-Stages.
- `point_count_after_downsample` ist die finale gespeicherte Punktzahl, also nach Symmetry Completion und finaler Konsolidierung.

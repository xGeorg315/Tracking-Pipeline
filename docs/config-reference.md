# Config-Referenz

Diese Seite beschreibt die YAML-Konfiguration der Pipeline auf Basis der aktuellen Dataclasses in `src/tracking_pipeline/config/models.py`.

## Ladeverhalten

- `tracking-pipeline run -c <preset>.yaml` laedt das angegebene YAML und merged es automatisch mit `base.yaml` im selben Verzeichnis.
- Der Merge ist ein **Deep Merge**: nur ueberschriebene Schluessel werden ersetzt.
- Relative `input.paths` werden relativ zur Config-Datei aufgeloest.
- Relative `classification.pointnext_root`, `classification.checkpoint_path` und `classification.model_cfg_path` werden zuerst relativ zur Config-Datei, dann relativ zu deren Parent und zuletzt relativ zum aktuellen Arbeitsverzeichnis aufgeloest.
- `input.paths` akzeptiert Dateien und Ordner. Ordner werden nicht rekursiv durchsucht; alle direkten `.pb`-Dateien werden lexikographisch nach Dateiname als eine Multi-File-Sequenz expandiert.
- Benchmark-Configs loesen `sequences` und `presets` ebenfalls relativ zur Manifestdatei auf.
- `benchmark.sequences` akzeptiert ebenfalls Dateien und Ordner. Ein Ordner repraesentiert genau eine Sequenz und wird erst pro Benchmark-Lauf zu seinen direkten `.pb`-Dateien expandiert.

## Unterstuetzte Werte

- `input.format`: `a42_pb`, `qb2_live`
- `clustering.algorithm`: `dbscan`, `euclidean_clustering`, `ground_removed_dbscan`, `hdbscan`, `voxel_grid_connected_components`, `range_image_connected_components`, `range_image_depth_jump`, `beam_neighbor_region_growing`
- `tracking.algorithm`: `euclidean_nn`, `kalman_nn`, `hungarian_kalman`
- `aggregation.algorithm`: `voxel_fusion`, `registration_voxel_fusion`, `weighted_voxel_fusion`, `occupancy_consensus_fusion`
- `aggregation.frame_selection_method`: `auto`, `all_track_frames`, `line_touch_last_k`, `keyframe_motion`, `length_coverage`, `quality_coverage`, `tail_coverage`, `center_diversity`, `max_extent`
- `aggregation.registration_backend`: `small_gicp`, `icp_point_to_plane`, `generalized_icp`, `feature_global_then_local`, `kiss_matcher`, `kiss_matcher_then_icp`
- `aggregation.registration_allowed_dofs`: `tx`, `ty`, `tz`, `roll`, `pitch`, `yaw`
- `aggregation.fusion_weight_mode`: `uniform`, `point_count`, `quality`

## Beispiel: Pipeline-Config

```yaml
input:
  paths:
    - ../data/sequence_dir
preprocessing:
  lane_box: [-2.10, 1.80, 4.0, 35.30, 0.12, 5.15]
clustering:
  algorithm: dbscan
tracking:
  algorithm: kalman_nn
aggregation:
  algorithm: voxel_fusion
output:
  root_dir: runs
```

## `input`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `paths` | kein Default | Eingabedateien oder Ordner; Ordner werden zu direkten `.pb`-Dateien in Namensreihenfolge expandiert; bei `qb2_live` darf die Liste leer sein |
| `format` | `a42_pb` | Reader-Auswahl (`a42_pb` fuer Dateien, `qb2_live` fuer Live-QB2 + MQTT) |
| `qb2_live` | `{}` | Live-Konfiguration fuer `qb2_live`; wird fuer `a42_pb` ignoriert |

### `input.qb2_live`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `sensor_name` | `""` | logischer Sensorname fuer Kalibrierung, Scans und GT-Labels |
| `ip` | `""` | QB2-IP/FQDN fuer `blickfeld_qb2.Channel` |
| `api_key` | `""` | Application-Key-Secret fuer den QB2-Zugriff |
| `api_key_file` | `""` | Pfad zu einer Datei mit dem QB2-API-Key; wird relativ zur Config aufgeloest |
| `mqtt.host` | `""` | MQTT-Broker fuer QB2-Objekt-/GT-Nachrichten |
| `mqtt.port` | `1883` | MQTT-Port |
| `mqtt.topic` | `""` | MQTT-Topic mit `states -> states -> trafficLane -> vehicles`-Payloads |
| `mqtt.keepalive` | `60` | MQTT-Keepalive in Sekunden |
| `max_frames` | `0` | optionales Live-Frame-Limit; `0` bedeutet unbegrenzt |
| `idle_timeout_sec` | `5.0` | Abbruch, wenn so lange keine neuen QB2-Rohframes eintreffen |
| `mqtt_drain_tolerance_sec` | `0.25` | kleine Zeit-Toleranz fuer leicht vorlaufende MQTT-Labels beim Anhaengen an Raw-Frames |
| `mqtt_max_pending_age_sec` | `3.0` | Pending-MQTT-Labels, die relativ zum aktuellen Raw-Frame aelter sind, werden aus der Queue geloescht |
| `mqtt_max_pending_labels` | `64` | maximales Pending-Budget fuer noch nicht angehaengte MQTT-Labels; bei Overflow bleiben die neuesten Labels erhalten |

Typische Live-Konfiguration:

```yaml
input:
  format: qb2_live
  paths: []
  qb2_live:
    sensor_name: class_qb2
    ip: 10.16.3.160
    api_key_file: apikey.txt
    mqtt:
      host: 10.16.3.111
      port: 1883
      topic: blickfeld/states_160
      keepalive: 60
    max_frames: 0
    idle_timeout_sec: 5.0
```

Wichtige Semantik:

- Wenn `paths: []` gesetzt ist, synthesisiert der Loader automatisch `qb2_live://<sensor_name>@<ip>` als stabile Run-Quelle.
- `qb2_live` wird in v1 nur von `tracking-pipeline run` unterstuetzt.
- `tracking-pipeline replay` und `tracking-pipeline benchmark` lehnen `qb2_live` bewusst frueh ab.
- Live-Runs aktualisieren `live_status.json` fortlaufend und schreiben Zwischenstaende fuer Objektliste, Tracks, Summary, GT-Matching und Aggregates bereits waehrend des Laufs.

## `runtime`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `cpu_cores` | `0` | Best-Effort-Limit fuer CPU-Kerne/Threads; `0` bedeutet keine explizite Begrenzung |

Beispiel:

```yaml
runtime:
  cpu_cores: 4
```

Hinweise:

- Auf Linux nutzt die Pipeline dafuer primaer CPU-Affinity auf dem Prozess.
- Zusaetzlich werden numerische Thread-Umgebungsvariablen sowie PyTorch-Thread-Limits best effort auf dieselbe Zahl gesetzt.

## `preprocessing`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `lane_box` | kein Default | `[x_min, x_max, y_min, y_max, z_min, z_max]` fuer Lane-Crop |
| `bootstrap_frames` | `10` | initiale Frames fuer Bootstrap-/Warmup-Logik |

## `clustering`

### Allgemein

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `algorithm` | `dbscan` | Clusterer-Auswahl |
| `eps` | `1.15` | DBSCAN-/Nachbarschaftsradius |
| `voxel_size` | `0.25` | Voxelgroesse fuer `voxel_grid_connected_components` |
| `min_points` | `20` | Mindestpunkte fuer DBSCAN/HDBSCAN |
| `vehicle_min_points` | `20` | Untergrenze fuer gueltige Fahrzeugcluster |
| `vehicle_max_points` | `10000` | Obergrenze fuer Fahrzeugcluster |

### Bodenentfernung

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `plane_distance_threshold` | `0.12` | RANSAC-Distanzschwelle fuer Bodenebene |
| `plane_ransac_n` | `3` | Anzahl Punkte pro RANSAC-Sample |
| `plane_num_iterations` | `120` | RANSAC-Iterationen |
| `ground_normal_z_min` | `0.75` | Mindest-`z` der Ebenennormale fuer Bodenannahme |

### HDBSCAN

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `hdbscan_min_cluster_size` | `20` | Mindestgroesse pro HDBSCAN-Cluster |
| `hdbscan_min_samples` | `10` | Nachbarschaftsparameter fuer HDBSCAN |

### Sensor-/Range-Image-Clusterung

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `sensor_range_min` | `0.0` | minimale gueltige Range |
| `sensor_range_max` | `120.0` | maximale gueltige Range |
| `sensor_depth_jump_ratio` | `0.08` | relative Tiefensprung-Schwelle |
| `sensor_depth_jump_abs` | `0.45` | absolute Tiefensprung-Schwelle |
| `sensor_min_component_size` | `8` | Mindestgroesse fuer Sensorraum-Komponenten |
| `sensor_neighbor_rows` | `1` | Nachbarschaft im Zeilenraum |
| `sensor_neighbor_cols` | `1` | Nachbarschaft im Spaltenraum |
| `sensor_ground_row_ignore` | `0` | wie viele untere Sensorzeilen ignoriert werden |

## `tracking`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `algorithm` | `kalman_nn` | Tracker-Auswahl |
| `max_dist` | `3.4` | Basis-Gating-Distanz fuer Zuordnung |
| `max_missed` | `12` | Frames bis ein Track als beendet gilt |
| `min_track_hits` | `4` | Mindestbeobachtungen fuer spaetere Aggregation |
| `sticky_extra_dist_per_missed` | `0.55` | zusaetzliche Toleranz pro verpasstem Frame |
| `sticky_max_dist` | `6.2` | Obergrenze fuer Sticky-Gating |
| `kf_init_var` | `5.0` | Initialvarianz des Kalman-Filters |
| `kf_process_var` | `0.08` | Prozessrauschen des Kalman-Filters |
| `kf_meas_var` | `0.60` | Messrauschen des Kalman-Filters |
| `association_size_weight` | `0.15` | Zusatzgewicht fuer Groessenunterschiede bei Zuordnung |

## `aggregation`

### Grundverhalten

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `algorithm` | `voxel_fusion` | Akkumulator-Auswahl |
| `symmetry_completion` | `false` | optionale lokale Symmetrievervollstaendigung |
| `motion_deskew` | `false` | objektrelativer intra-scan Deskew fuer elongierte Tracks anhand von `timestamp_offset` |
| `truncate_after_lane_end_touch` | `false` | verwirft alle Folgeframes nach erstem Touch am Lane-Ende (`frame_selection_line_axis`-Min-Seite) |
| `frame_selection_method` | `auto` | Strategie zur Chunk-/Frame-Auswahl: `all_track_frames`, `line_touch_last_k`, `keyframe_motion`, `length_coverage`, `quality_coverage`, `tail_coverage`, `center_diversity`, `max_extent`, `auto` |
| `use_all_frames` | `true` | alle Track-Frames verwenden statt nur Auswahl |
| `top_k_frames` | `10` | Limit fuer `line_touch_last_k` / Top-K-Auswahl |
| `keyframe_keep` | `8` | Anzahl Keyframes bei Keyframe-Auswahl |
| `frame_selection_line_axis` | `y` | Lane-Laengsachse |
| `frame_selection_line_ratio` | `0.10` | Position der Selektionslinie in Lane-Richtung |
| `frame_selection_touch_margin` | `0.12` | Toleranz fuer Line-Touch-Selektion |
| `frame_downsample_voxel` | `0.07` | fruehes Downsampling pro Chunk |

### Chunk-Qualitaet und Konsistenz

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `chunk_quality_filter` | `true` | filtert schwache Randbeobachtungen vor der Selektion |
| `chunk_min_points_ratio_to_peak` | `0.40` | Mindestverhaeltnis zur Peak-Punktzahl |
| `chunk_min_extent_ratio_to_peak` | `0.35` | Mindestverhaeltnis zur Peak-Ausdehnung |
| `chunk_min_segment_length` | `4` | minimale zusammenhaengende Segmentlaenge |
| `shape_consistency_filter` | `false` | filtert Formausreisser nach der Selektion |
| `shape_consistency_max_extent_ratio` | `2.0` | erlaubte Extent-Abweichung fuer Shape-Consistency |

### Registrierung

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `registration_backend` | `small_gicp` | Backend fuer `registration_voxel_fusion` (`kiss_matcher_then_icp` = KISS-Matcher Initial-Guess plus ICP-Refinement) |
| `registration_max_corr_dist` | `0.95` | maximale Korrespondenzdistanz |
| `registration_max_iter` | `80` | Iterationslimit fuer lokales Alignment |
| `registration_min_fitness` | `0.25` | Mindestfitness fuer akzeptierte Registrierung |
| `registration_max_translation` | `3.2` | Obergrenze fuer akzeptierte Translation |
| `registration_allowed_dofs` | `[tx, ty, tz, roll, pitch, yaw]` | erlaubte Freiheitsgrade fuer die angewendete Registrierungs-Transformation |
| `registration_max_dof_change` | `{}` | optionale per-DOF-Klemmung relativ zur Identitaet; `tx/ty/tz` in Metern, `roll/pitch/yaw` in Grad |
| `enable_registration_underfill_fallback` | `false` | faellt bei zu wenigen behaltenen Registration-Chunks auf die unregistrierten selektierten Chunks zurueck |
| `registration_min_kept_chunks` | `4` | Mindestanzahl an Registration-Chunks vor dem optionalen Underfill-Fallback |
| `global_registration_voxel` | `0.12` | Downsampling fuer globales Feature-Matching |

### Fusion und Save-Gating

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `fusion_voxel_size` | `0.05` | Voxelgroesse der eigentlichen Fusion |
| `fusion_min_observations` | `1` | Mindestanzahl Beobachtungen pro Fusionsvoxel |
| `fusion_weight_mode` | `point_count` | Gewichtung bei `weighted_voxel_fusion` |
| `consensus_ratio` | `0.35` | Konsensschwelle bei `occupancy_consensus_fusion` |
| `min_track_quality_for_save` | `0.0` | Mindestqualitaet fuer normale Tracks |
| `min_saved_aggregate_points` | `180` | Untergrenze fuer gespeicherte Aggregate |
| `enable_confidence_point_cap` | `false` | kappt das finale gespeicherte Aggregate optional auf die confidentesten Punkte |
| `confidence_point_cap_max_points` | `2048` | maximales Punktbudget fuer das finale Aggregate bei aktivem Confidence-Cap |
| `confidence_point_cap_bins` | `16` | Anzahl Longitudinal-Bins fuer die verteilte Confidence-Auswahl |
| `enable_post_filter_stat_outlier_removal` | `true` | schaltet Statistical Outlier Removal im normalen Post-Filter und im Long-Vehicle-Merge explizit ein oder aus |
| `aggregate_voxel` | `0.06` | finales Downsampling nach dem Post-Filter |

### Long-Vehicle-Modus

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `long_vehicle_mode` | `false` | schaltet Long-Vehicle-Logik ein |
| `long_vehicle_length_threshold` | `4.5` | Schwellwert fuer Long-Vehicle-Klassifikation |
| `length_coverage_bins` | `10` | Bins fuer laengenbasierte Frame-Selektion |
| `min_track_quality_for_save_long_vehicle` | `0.0` | Mindestqualitaet fuer Long Vehicles |
| `enable_tail_bridge` | `true` | schaltet Tail-Bridge komplett ein oder aus; bei `false` entfaellt auch die teure Komponentenbildung in diesem Schritt |
| `tail_bridge_longitudinal_gap_max` | `1.5` | maximaler Laengsabstand fuer Tail-Bridge |
| `tail_bridge_lateral_gap_max` | `0.8` | maximaler lateraler Abstand fuer Tail-Bridge |
| `tail_bridge_vertical_gap_max` | `0.5` | maximaler Hoehenabstand fuer Tail-Bridge |

### Post-Filter

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `post_filter_stat_nb_neighbors` | `12` | Nachbarn fuer Statistical Outlier Removal, falls aktiviert |
| `post_filter_stat_std_ratio` | `2.3` | Streuungsfaktor fuer Outlier Removal |

## `postprocessing`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enable_tracklet_stitching` | `false` | aktiviert Tracklet-Stitching |
| `stitching_max_gap` | `4` | maximaler zeitlicher Gap fuer Stitching |
| `stitching_max_center_dist` | `2.5` | maximaler Mittelpunktabstand fuer Stitching |
| `enable_articulated_vehicle_merge` | `false` | merged stabile Front-/Rear-Teiltracks von Zugfahrzeug und Anhaenger im finalen Output |
| `articulated_gap_eval_window_frames` | `5` | bewertet den Hitch-Gap ueber die letzten gemeinsamen Frames; bei weniger als 3 Frames faellt der Merge auf die volle Ueberlappung zurueck |
| `articulated_min_overlap_frames` | `4` | Mindestueberlappung in Frames fuer Trailer-Merge |
| `articulated_min_overlap_ratio` | `0.5` | Mindestueberlappung als Anteil fuer Trailer-Merge |
| `articulated_max_lateral_offset` | `0.9` | maximaler mittlerer lateraler Offset zwischen Zugfahrzeug und Anhaenger |
| `articulated_max_vertical_offset` | `0.6` | maximaler mittlerer Hoehenoffset zwischen Zugfahrzeug und Anhaenger |
| `articulated_max_hitch_gap` | `2.5` | maximaler mittlerer Hitch-/Rear-Gap entlang der Laengsachse |
| `articulated_max_hitch_gap_std` | `0.6` | maximal zulaessige Gap-Schwankung fuer Trailer-Merge |
| `articulated_max_speed_delta` | `0.5` | maximaler Unterschied der Longitudinalgeschwindigkeit |
| `articulated_min_combined_length` | `6.5` | minimale kombinierte Laengsausdehnung fuer artikulierte Fahrzeuge |
| `enable_co_moving_track_merge` | `false` | aktiviert Merge paralleler Co-Moving-Tracks |
| `parallel_merge_max_lateral_offset` | `0.8` | maximaler lateraler Offset fuer Merge |
| `parallel_merge_max_longitudinal_gap` | `4.0` | maximaler Laengsgap fuer Merge |
| `parallel_merge_min_overlap_frames` | `5` | Mindestueberlappung in Frames |
| `parallel_merge_min_overlap_ratio` | `0.6` | Mindestueberlappung als Anteil |
| `enable_trajectory_smoothing` | `false` | aktiviert Glaettung der Track-Zentren |
| `smoothing_window` | `3` | Fensterbreite fuer Glaettung |
| `enable_track_quality_scoring` | `true` | berechnet Track-Qualitaet vor der Aggregation |

## `classification`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enabled` | `false` | schaltet die optionale Objektklassifikation nach der finalen Aggregation ein |
| `backend` | `pointnext` | Klassifikationsbackend; aktuell nur PointNeXt |
| `pointnext_root` | `PointNeXt` | Root des eingecheckten PointNeXt-Repos; `openpoints` muss darunter importierbar sein |
| `checkpoint_path` | `ckpt/bestckpt.pth` | Checkpoint fuer die Inferenz |
| `model_cfg_path` | `PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml` | PointNeXt-Model-Config fuer Build und `num_points` |
| `class_names` | `[]` | fixe Klassenliste; beim Laden und in der PointNeXt-Initialisierung wird sie auf `sorted(class_dir.name)` normalisiert |
| `device` | `auto` | `auto`, `cpu`, `cuda` oder `mps`; `auto` nutzt zuerst CUDA, sonst MPS, sonst CPU |

Wichtige Einschraenkungen:

- Die aktuelle Inferenz verwendet nur `xyz` und erwartet deshalb `model.encoder_args.in_channels == 3`.
- `model.extra_global_channels` muss `0` sein; zusaetzliche globale Features aus dem Trainings-Dataloader werden in v1 bewusst nicht unterstuetzt.
- Wenn `classification.enabled=true` und `PointNeXt/openpoints` leer oder nicht importierbar ist, bricht die Pipeline frueh mit einer klaren Fehlermeldung ab.
- `device: mps` benoetigt eine PyTorch-Version, die MPS fuer deine aktuelle macOS-Version korrekt freischaltet; bei neueren macOS-Releases kann dafuer ein Torch-Upgrade noetig sein.

## `class_normalization`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enabled` | `false` | aktiviert die Klassen-Normalisierung auf eine gemeinsame Zieltaxonomie |
| `aliases` | `{}` | Mapping von rohen Klassenamen auf kanonische Zielnamen; Lookup ist case-insensitive und trimmt Whitespace |

Typischer Einsatz:

- Modellklassen wie `PKW`, `LKW` oder `Transporter` auf TLS-Klassen mappen
- GT-Labels wie `car`, `truck` oder `van` auf dieselbe TLS-Taxonomie mappen
- dadurch Statistik, Viewer und Exporte im selben Klassenraum halten

Wichtige Semantik:

- die Normalisierung schreibt bestehende Felder wie `predicted_class_name` und `gt_obj_class` bewusst um
- unbekannte Klassen bleiben unveraendert
- `TLS_VEHICLE_OTHER` wird nicht automatisch aus unbekannten Prediction-Klassen erzeugt; dafuer braucht es einen expliziten Alias

## `output`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `mode` | `run` | exklusiver Output-Modus: `run` fuer klassischen Run-Ordner, `dataset` fuer den globalen Dataset-Baum |
| `root_dir` | `runs` | Zielverzeichnis fuer Run-Artefakte bei `mode: run` |
| `dataset_root_dir` | `dataset` | Zielverzeichnis fuer den globalen Dataset-Baum bei `mode: dataset` |
| `save_world` | `false` | speichert Aggregate in Welt- statt Lokalkoordinaten |
| `save_aggregate_intensity` | `false` | schreibt range-korrigierte Reflectivity als PCD-Feld `reflectivity` mit |
| `require_track_exit` | `true` | speichert nur Tracks, die die Lane-Box verlassen haben |
| `track_exit_edge_margin` | `0.9` | Offset der Exit-Linie von der Min-Seite der Lane-Laengsachse; der letzte Track-Center muss diese Linie passiert haben |
| `statistics_enabled` | `true` | schreibt Run-/Live-Statistiken wie `_stats`, Summary, Tracks, Tracker-Debug, Track-Outcomes, Class-Stats und Performance; `false` laesst Core-Artefakte wie Aggregate und GT-Matching aktiv |
| `final_full_recompute` | `true` | fuehrt am Run-Ende den vollstaendigen Recompute ueber alle Tracks inkl. GT-Matching aus; `false` beendet den Run schneller und nutzt den inkrementellen Live-Stand als groben Endstand |
| `live_object_list_flush_interval_sec` | `1.0` | Mindestabstand zwischen zwei Live-Object-List-Flushes; `0` schreibt sofort bei jeder Aenderung |
| `live_artifact_flush_interval_sec` | `2.0` | Mindestabstand zwischen zwei inkrementellen Live-Snapshot-Flushes fuer Tracks/Outcomes/Summary; `0` schreibt sofort bei jedem neuen beendeten Track |
| `live_tracker_debug_flush_interval_sec` | `10.0` | Mindestabstand fuer Live-Tracker-Debug-Snapshots in den Stats-Dateien; `0` schreibt bei jedem Live-Snapshot |

Hinweise:

- Es wird immer genau eines geschrieben: entweder der Run-Ordner oder der Dataset-Baum.
- `benchmark` lehnt `output.mode: dataset` bewusst ab.
- Im `dataset`-Modus liegen Tagesstatistiken unter `dataset/_stats/YYYY-MM-DD/<run_id>/`, sofern `statistics_enabled: true` gesetzt ist.
- Mit `statistics_enabled: false` wird kein `_stats`-Baum erzeugt; Live-Aggregate und GT-Matching-Dataset-Samples werden weiter geschrieben.
- `tracking-pipeline live-view -c <config>` unterstuetzt in v1 nur `output.mode: dataset` und liest die bestehenden Snapshot-Dateien read-only aus diesem Baum.
- `tracking-pipeline live-web -c <config> [--host <host>] [--port <port>]` bleibt ein Snapshot-Fallback fuer denselben Dataset-Pfad; der echte Live-Raw-PCD-Webviewer wird direkt aus `tracking-pipeline run` heraus ueber `visualization.live_web_*` gestartet.
- Die drei `live_*_flush_interval_sec`-Schalter beeinflussen nur laufende Live-Snapshots; mit `statistics_enabled: true` wird der finale Stats-Output am Ende weiterhin vollstaendig geschrieben.

## `visualization`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enabled` | `true` | aktiviert Replay-Visualisierung |
| `color_by_intensity` | `false` | faerbt Lane/Cluster/Aggregate nach range-korrigierter Reflectivity; nur fuer die Anzeige robust normalisiert |
| `show_full_frame_pcd` | `false` | blendet die komplette rohe Frame-Punktwolke als Hintergrund-Layer im Replay ein |
| `show_tracker_debug` | `false` | zeigt Tracker-Predictions, Match-/Miss-/Spawn-Overlay und HUD im Replay; dient auch als Default-Toggle fuer `live-view` |
| `show_track_outcome_debug` | `false` | zeigt Save-/Skip-Beacons, Failure-Tails und Outcome-HUD fuer finalisierte Tracks; dient auch als Default-Toggle fuer `live-view` |
| `show_articulated_merge_debug` | `false` | zeigt Trailer-Merge-Paare, Tail-/Full-Gap-HUD und Outcome-Beacons im Replay; per Taste `M` umschaltbar |
| `live_web_enabled` | `false` | startet fuer `input.format: qb2_live` einen eingebetteten headless Browser-Viewer direkt im laufenden `run`-Prozess |
| `live_web_host` | `0.0.0.0` | Bind-Host fuer den eingebetteten Browser-Viewer |
| `live_web_port` | `8765` | TCP-Port fuer den eingebetteten Browser-Viewer |
| `live_web_history_sec` | `0.8` | Rolling-History-Fenster fuer den Browser-Viewer |
| `live_web_point_source` | `lane` | Punktquelle fuer den eingebetteten Browser-Viewer; `lane` streamt nur Lane-Punkte, `all` die komplette Frame-Punktwolke |
| `max_points` | `120000` | Punktlimit fuer den Viewer |
| `max_cluster_points` | `15000` | Punktlimit pro Cluster im Viewer |
| `max_assoc_dist` | `4.2` | Darstellungsdistanz fuer Assoziationshilfen |

Hinweise:

- `live-view` ist ein separater Open3D-Prozess fuer den aktuellen Snapshot-Zustand eines laufenden `dataset`-Runs, nicht das normale Replay.
- V1 von `live-view` zeigt bewusst keine volle Rohpunktwolke, keine Aggregate-PCDs und keine Playback-Timeline, sondern Lane-Box, Exit-Linie, letzten Tracker-Snapshot, Outcome-Beacons und HUD.
- Wenn `visualization.live_web_enabled: true` aktiv ist und `input.format: qb2_live` laeuft, startet `tracking-pipeline run` einen eingebetteten Browser-Viewer fuer die rohe Live-Punktwolke direkt im Pipeline-Prozess.
- Dieser Live-Webviewer streamt pro Frame je nach `live_web_point_source` entweder nur `lane_points` oder die komplette Frame-Punktwolke als `points_xyz`, dazu Tracker-Overlay und die zuletzt bekannten `track_outcomes` in den Browser.
- Live-Frames werden im eingebetteten Browser-Viewer immer als Rolling Window gehalten; Full-Run-Replay erfolgt ueber gespeicherte Artefakte, nicht ueber den Browser-Buffer.
- `live_web_history_sec` bestimmt, wie lange Frames im direkten Rohframe-Stream gehalten werden.
- `max_points` dient in diesem Pfad als serverseitiges Punktbudget pro Live-Frame.
- Der CLI-Befehl `tracking-pipeline live-web ...` bleibt als read-only Snapshot-Fallback erhalten und nutzt weiterhin die bereits geschriebenen Dataset-Stats statt den direkten Rohframe-Stream.

## Benchmark-Manifest

```yaml
sequences:
  - ../data/sequence_dir
presets:
  - ./kalman_voxel.yaml
  - ./kalman_small_gicp.yaml
output_root: benchmarks
name: curated_real
warmup_runs: 1
measure_runs: 1
```

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `sequences` | kein Default | Eingabedateien oder Ordner; jeder Ordner ist genau eine Benchmark-Sequenz aus seinen direkten `.pb`-Dateien in Namensreihenfolge |
| `presets` | kein Default | Presets, die gegeneinander verglichen werden |
| `output_root` | `benchmarks` | Zielverzeichnis fuer Benchmark-Artefakte |
| `name` | `curated_proxy` | Suffix im Benchmark-Ordnernamen |
| `warmup_runs` | `1` | Warmup-Laeufe pro Preset/Sequenz |
| `measure_runs` | `1` | gemessene Laeufe pro Preset/Sequenz; fuer wiederholte Laufzeitmessung explizit groesser als `1` setzen |

## Typische Presets

| Preset | Wofuer geeignet |
| --- | --- |
| `kalman_voxel.yaml` | schneller Start ohne Registrierung |
| `kalman_small_gicp.yaml` | registrierungsbasierte Fusion mit optionalem `small_gicp` |
| `kalman_generalized_icp.yaml` | Vergleich von Open3D-GICP gegen `small_gicp` |
| `kalman_feature_global_then_local.yaml` | schwierigere Initiallagen mit globalem Vorab-Alignment |
| `hungarian_weighted.yaml` | globales Matching plus gewichtete Fusion |
| `long_vehicle_*.yaml` | Presets fuer laengere Fahrzeuge und Tail-Bridge-Logik |

## Praktische Hinweise

- Fuer normale Einstiege ist `kalman_voxel.yaml` die einfachste stabile Basis.
- `registration_voxel_fusion` lohnt sich nur, wenn Chunks ohne Registrierung sichtbar versetzt bleiben.
- `enable_articulated_vehicle_merge` wirkt nur auf finale Tracks und Aggregate; im normalen Replay-Layer bleiben die Originaltracks sichtbar. Optional kann `show_articulated_merge_debug` zusaetzlich das Merge-Debug-Overlay einblenden.
- `motion_deskew`, `symmetry_completion` und `save_aggregate_intensity` sind additive Features auf dem finalen Aggregate-Output.
- Wenn ein Preset nicht explizit alle Werte setzt, kommen sie aus `configs/base.yaml`.

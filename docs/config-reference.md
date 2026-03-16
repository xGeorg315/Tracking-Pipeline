# Config-Referenz

Diese Seite beschreibt die YAML-Konfiguration der Pipeline auf Basis der aktuellen Dataclasses in `src/tracking_pipeline/config/models.py`.

## Ladeverhalten

- `tracking-pipeline run -c <preset>.yaml` laedt das angegebene YAML und merged es automatisch mit `base.yaml` im selben Verzeichnis.
- Der Merge ist ein **Deep Merge**: nur ueberschriebene Schluessel werden ersetzt.
- Relative `input.paths` werden relativ zur Config-Datei aufgeloest.
- Benchmark-Configs loesen `sequences` und `presets` ebenfalls relativ zur Manifestdatei auf.

## Unterstuetzte Werte

- `input.format`: `a42_pb`
- `clustering.algorithm`: `dbscan`, `euclidean_clustering`, `ground_removed_dbscan`, `hdbscan`, `range_image_connected_components`, `range_image_depth_jump`, `beam_neighbor_region_growing`
- `tracking.algorithm`: `euclidean_nn`, `kalman_nn`, `hungarian_kalman`
- `aggregation.algorithm`: `voxel_fusion`, `registration_voxel_fusion`, `weighted_voxel_fusion`, `occupancy_consensus_fusion`
- `aggregation.frame_selection_method`: `auto`, `all_track_frames`, `line_touch_last_k`, `keyframe_motion`, `length_coverage`
- `aggregation.registration_backend`: `small_gicp`, `icp_point_to_plane`, `generalized_icp`, `feature_global_then_local`
- `aggregation.fusion_weight_mode`: `uniform`, `point_count`, `quality`

## Beispiel: Pipeline-Config

```yaml
input:
  paths:
    - ../data/3.pb
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
| `paths` | kein Default | Eingabedateien oder Sequenzen |
| `format` | `a42_pb` | Reader-Auswahl |

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
| `frame_selection_method` | `auto` | Strategie zur Chunk-/Frame-Auswahl |
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
| `registration_backend` | `small_gicp` | Backend fuer `registration_voxel_fusion` |
| `registration_max_corr_dist` | `0.95` | maximale Korrespondenzdistanz |
| `registration_max_iter` | `80` | Iterationslimit fuer lokales Alignment |
| `registration_min_fitness` | `0.25` | Mindestfitness fuer akzeptierte Registrierung |
| `registration_max_translation` | `3.2` | Obergrenze fuer akzeptierte Translation |
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
| `aggregate_voxel` | `0.06` | finales Downsampling nach dem Post-Filter |

### Long-Vehicle-Modus

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `long_vehicle_mode` | `false` | schaltet Long-Vehicle-Logik ein |
| `long_vehicle_length_threshold` | `4.5` | Schwellwert fuer Long-Vehicle-Klassifikation |
| `length_coverage_bins` | `10` | Bins fuer laengenbasierte Frame-Selektion |
| `min_track_quality_for_save_long_vehicle` | `0.0` | Mindestqualitaet fuer Long Vehicles |
| `tail_bridge_longitudinal_gap_max` | `1.5` | maximaler Laengsabstand fuer Tail-Bridge |
| `tail_bridge_lateral_gap_max` | `0.8` | maximaler lateraler Abstand fuer Tail-Bridge |
| `tail_bridge_vertical_gap_max` | `0.5` | maximaler Hoehenabstand fuer Tail-Bridge |

### Post-Filter

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `post_filter_stat_nb_neighbors` | `12` | Nachbarn fuer Statistical Outlier Removal |
| `post_filter_stat_std_ratio` | `2.3` | Streuungsfaktor fuer Outlier Removal |

## `postprocessing`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enable_tracklet_stitching` | `false` | aktiviert Tracklet-Stitching |
| `stitching_max_gap` | `4` | maximaler zeitlicher Gap fuer Stitching |
| `stitching_max_center_dist` | `2.5` | maximaler Mittelpunktabstand fuer Stitching |
| `enable_co_moving_track_merge` | `false` | aktiviert Merge paralleler Co-Moving-Tracks |
| `parallel_merge_max_lateral_offset` | `0.8` | maximaler lateraler Offset fuer Merge |
| `parallel_merge_max_longitudinal_gap` | `4.0` | maximaler Laengsgap fuer Merge |
| `parallel_merge_min_overlap_frames` | `5` | Mindestueberlappung in Frames |
| `parallel_merge_min_overlap_ratio` | `0.6` | Mindestueberlappung als Anteil |
| `enable_trajectory_smoothing` | `false` | aktiviert Glaettung der Track-Zentren |
| `smoothing_window` | `3` | Fensterbreite fuer Glaettung |
| `enable_track_quality_scoring` | `true` | berechnet Track-Qualitaet vor der Aggregation |

## `output`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `root_dir` | `runs` | Zielverzeichnis fuer Run-Artefakte |
| `save_world` | `false` | speichert Aggregate in Welt- statt Lokalkoordinaten |
| `save_aggregate_intensity` | `false` | schreibt `intensity` als PCD-Feld mit |
| `require_track_exit` | `true` | speichert nur Tracks, die die Lane-Box verlassen haben |
| `track_exit_edge_margin` | `0.9` | Randtoleranz fuer Track-Exit-Check |

## `visualization`

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `enabled` | `true` | aktiviert Replay-Visualisierung |
| `color_by_intensity` | `false` | faerbt Lane/Cluster/Aggregate nach Intensitaet |
| `max_points` | `120000` | Punktlimit fuer den Viewer |
| `max_cluster_points` | `15000` | Punktlimit pro Cluster im Viewer |
| `max_assoc_dist` | `4.2` | Darstellungsdistanz fuer Assoziationshilfen |

## Benchmark-Manifest

```yaml
sequences:
  - ../data/3.pb
presets:
  - ./kalman_voxel.yaml
  - ./kalman_small_gicp.yaml
output_root: benchmarks
name: curated_real
warmup_runs: 1
measure_runs: 3
```

| Feld | Default | Bedeutung |
| --- | --- | --- |
| `sequences` | kein Default | Eingabesequenzen fuer den Benchmark |
| `presets` | kein Default | Presets, die gegeneinander verglichen werden |
| `output_root` | `benchmarks` | Zielverzeichnis fuer Benchmark-Artefakte |
| `name` | `curated_proxy` | Suffix im Benchmark-Ordnernamen |
| `warmup_runs` | `1` | Warmup-Laeufe pro Preset/Sequenz |
| `measure_runs` | `3` | gemessene Laeufe pro Preset/Sequenz |

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
- `symmetry_completion` und `save_aggregate_intensity` sind additive Features auf dem finalen Aggregate-Output.
- Wenn ein Preset nicht explizit alle Werte setzt, kommen sie aus `configs/base.yaml`.

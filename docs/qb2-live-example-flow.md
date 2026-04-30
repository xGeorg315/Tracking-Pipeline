# QB2 Live Example: Pipeline-Verlauf

Diese Seite beschreibt den Ablauf der Config `configs/qb2_live_example.yaml`.
Die Config wird bei `tracking-pipeline run -c configs/qb2_live_example.yaml`
automatisch mit `configs/base.yaml` gemerged. Dadurch kommen unter anderem
`preprocessing.lane_box`, der Default-Clusterer `dbscan`, PointNeXt-Defaults,
Class-Normalization und mehrere Output-Gates aus `base.yaml`.

## Kurzuebersicht

```text
QB2 raw stream + MQTT states
  -> Lane-Crop
  -> DBSCAN-Detections
  -> Kalman-NN-Tracking
  -> beendete Tracks
  -> articulated merge + quality scoring
  -> Keyframe-Auswahl
  -> ICP point-to-plane Registrierung
  -> Voxel-Fusion + Filter + Symmetry Completion
  -> PointNeXt-Klassifikation
  -> GT-Matching gegen MQTT-Objekte
  -> Dataset-Samples
```

## 1. Live-Input

`input.format: qb2_live` aktiviert den `QB2LiveReader`. Er verbindet sich mit
dem QB2-Sensor ueber die konfigurierte IP und liest Rohframes aus dem
QB2-PointCloud-Service. Parallel startet er einen MQTT-Client, der Object-/
GT-States vom konfigurierten MQTT-Topic liest.

Wichtige Werte aus der Config:

- `sensor_name: class_qb2`
- `ip: 10.16.3.160`
- `api_key_file: apikey.txt`
- `mqtt.host: 10.16.3.111`
- `mqtt.topic: blickfeld/states_160`
- `idle_timeout_sec: 30.0`
- `mqtt_drain_tolerance_sec: 5.0`
- `mqtt_max_pending_age_sec: 20.0`
- `mqtt_max_pending_labels: 300`

Wenn `input.paths: []` gesetzt ist, erzeugt der Loader automatisch einen
stabilen Quellnamen:

```text
qb2_live://class_qb2@10.16.3.160
```

Pro Rohframe werden die QB2-Punkte in `FrameData` umgewandelt. MQTT-Objekte
werden nach Timestamp gepuffert und beim naechsten passenden Frame in
`frame.object_labels` einsortiert. Die zusaetzliche Drain-Toleranz von 5 s
hilft, kleine Laufzeitunterschiede zwischen Rohstream und MQTT auszugleichen.

## 2. Lane-Crop und Clustering

Der Clusterer kommt aus `base.yaml`, sofern die Live-Config ihn nicht
ueberschreibt:

```yaml
clustering:
  algorithm: dbscan
  eps: 0.90
  min_points: 8
  vehicle_min_points: 20
  vehicle_max_points: 10000
```

Zuerst werden die Rohpunkte mit `preprocessing.lane_box` auf die relevante
Lane-Region beschnitten. Danach laeuft DBSCAN auf diesen Lane-Punkten. Jeder
akzeptierte Cluster wird zu einer `Detection` mit:

- Punktwolke des Clusters
- Center als Mittelwert der Punkte
- Bounding-Box-Min/Max
- optionaler Intensitaet
- Punktanzahl und Cluster-Metriken

Cluster mit weniger als `vehicle_min_points` oder mehr als
`vehicle_max_points` werden verworfen.

## 3. Tracking

Die Config nutzt:

```yaml
tracking:
  algorithm: kalman_nn
  max_dist: 7.5
  max_missed: 2
  min_track_hits: 6
  sticky_extra_dist_per_missed: 1.0
  sticky_max_dist: 12.0
  kf_process_var: 0.8
  kf_meas_var: 1.2
  association_size_weight: 0.05
```

Der `KalmanNNTracker` fuehrt pro bestehendem Track eine Kalman-Prediction aus
und ordnet neue Detections greedy nach Kosten zu. Die Kosten bestehen aus
Center-Distanz plus einem kleinen Extent-Unterschied:

```text
cost = distance(predicted_center, detection_center)
       + association_size_weight * normalized_extent_delta
```

Ein Match ist nur erlaubt, wenn die Kosten im Gate liegen:

```text
gate = min(sticky_max_dist, max_dist + sticky_extra_dist_per_missed * missed)
```

Bei dieser Config ist das anfangs `7.5 m`, steigt bei verpassten Frames um
`1.0 m` und ist bei `12.0 m` gedeckelt.

Wenn ein Track mehr als `max_missed: 2` Frames nicht gematcht wurde, wird er
beendet und fuer die Aggregation freigegeben.

## 4. Live-Snapshots und Finalisierung

Die Live-Config setzt:

```yaml
output:
  mode: dataset
  statistics_enabled: false
  final_full_recompute: false
  live_artifact_flush_interval_sec: 20.0
```

Damit verarbeitet die Pipeline beendete Tracks inkrementell. Sie wartet also
nicht bis zum Ende, um alle Tracks komplett neu zu berechnen, sondern haelt
laufend einen Snapshot der bereits fertig verarbeiteten Tracks/Aggregate.
Dataset-Artefakte werden in Intervallen geschrieben. Da
`statistics_enabled: false` gesetzt ist, werden keine `_stats`-Debugdateien
wie `summary.json`, `tracks.jsonl` oder `tracker_debug.jsonl` geschrieben.
Die eigentlichen Dataset-Samples und GT-Matching-Ausgaben bleiben aktiv.

## 5. Aggregation: Track zu Objektpunktwolke

Dieser Abschnitt entspricht dem wichtigsten Teil nach dem Tracking. Die Config
nutzt:

```yaml
aggregation:
  algorithm: registration_voxel_fusion
  registration_backend: icp_point_to_plane
  frame_selection_method: keyframe_motion
  keyframe_keep: 8
  symmetry_completion: true
  enable_registration_underfill_fallback: true
  enable_tail_bridge: false
  enable_post_filter_stat_outlier_removal: true
  registration_min_kept_chunks: 2
  registration_max_iter: 30
  registration_max_translation: 3.5
  registration_min_fitness: 0.3
  registration_max_corr_dist: 0.5
  shape_consistency_filter: false
  truncate_after_lane_end_touch: true
  enable_confidence_point_cap: false
```

### 5.1 Harte Gates

Der Accumulator bekommt einen beendeten Track und prueft zuerst:

- Hat der Track mindestens `tracking.min_track_hits: 6` Treffer?
- Hat der Track die Lane verlassen?
- Ist der Track bereits missed oder beendet?

Das Lane-Exit-Gate kommt aus `base.yaml`:

```yaml
output:
  require_track_exit: true
  track_exit_edge_margin: 5
aggregation:
  frame_selection_line_axis: y
```

Die Exit-Linie liegt auf der longitudinalen `y`-Achse bei:

```text
lane_box.y_min + track_exit_edge_margin
```

Mit der Base-Lane-Box ist das ungefaehr `4.0 + 5.0 = 9.0`. Ein Track wird
nur gespeichert, wenn sein letzter Center diese Linie passiert hat und der
Track nicht mehr aktiv gematcht wird.

### 5.2 Chunks sammeln und beschneiden

Ein Track speichert pro Beobachtung einen Chunk:

- `track.world_points`: Punkte des Clusters im jeweiligen Frame
- `track.centers`: Track-Center je Frame
- `track.frame_ids`: Frame-IDs
- optional Intensitaet und Punkt-Timestamps

Mit `truncate_after_lane_end_touch: true` sucht die Pipeline den ersten Frame,
in dem die Track-Punkte das Lane-Ende beruehren. Alles nach diesem Frame wird
abgeschnitten. Das verhindert, dass stark abgeschnittene Nachlauf-Beobachtungen
beim Herausfahren die finale Fahrzeugwolke verschlechtern.

### 5.3 Chunk-Qualitaetsfilter

Aus `base.yaml` ist der Chunk-Qualitaetsfilter aktiv:

```yaml
chunk_quality_filter: true
chunk_min_points_ratio_to_peak: 0.60
chunk_min_extent_ratio_to_peak: 0.55
chunk_min_segment_length: 4
```

Die Pipeline bewertet jeden Chunk ueber Punktanzahl und raeumliche Ausdehnung.
Sie sucht den Peak und behaelt bevorzugt einen zusammenhaengenden Bereich von
guten Chunks. Zu kleine oder stark unterfuellte Chunks fallen heraus. Falls
kein guter Bereich gefunden wird, wird um den besten Chunk herum erweitert.

### 5.4 Keyframe-Auswahl

Die Live-Config setzt:

```yaml
frame_selection_method: keyframe_motion
keyframe_keep: 8
```

`keyframe_motion` waehlt bis zu 8 informative Frames. Dafuer werden die
Bewegungen der Track-Center ausgewertet. Frames mit hoher Bewegung sind
interessant, weil sie meist neue Fahrzeugbereiche zeigen. Anfang und Ende
werden ebenfalls bevorzugt erhalten.

Bei langen Fahrzeugen kann die Pipeline intern auf eine laengenbasierte
Coverage-Strategie wechseln, damit Vorderteil, Mitte und Heck besser
abgedeckt werden.

## 6. ICP-Registrierung

Da `algorithm: registration_voxel_fusion` gesetzt ist, werden die ausgewaehlten
Chunks vor der Voxel-Fusion registriert. Das Backend ist:

```yaml
registration_backend: icp_point_to_plane
```

Vor der Registrierung passiert noch Vorbereitung:

- optional Motion Deskew aus `base.yaml`, falls Punkt-Timestamps vorhanden sind
- Umrechnung in lokale Objektkoordinaten, weil `output.save_world: false`
- Downsampling pro Frame mit `frame_downsample_voxel: 0.07` aus `base.yaml`

Der ICP-Ablauf ist inkrementell:

1. Chunk 0 wird als Startmodell genommen.
2. Chunk 1 wird gegen das aktuelle Modell registriert.
3. Bei erfolgreichem Alignment wird Chunk 1 transformiert und ins Modell
   integriert.
4. Das Modell wird wieder downsampled.
5. Der naechste Chunk wird gegen dieses gewachsene Modell registriert.
6. Das wiederholt sich fuer alle ausgewaehlten Chunks.

`icp_point_to_plane` nutzt Open3D-ICP mit Normalenschaetzung und
`TransformationEstimationPointToPlane`.

Ein Alignment wird akzeptiert, wenn:

- `fitness >= registration_min_fitness`, hier `0.3`
- die Transformation numerisch gueltig ist
- die Translation hoechstens `registration_max_translation`, hier `3.5 m`, ist
- die Korrespondenzen innerhalb `registration_max_corr_dist`, hier `0.5 m`,
  liegen

Akzeptierte Chunks bekommen ein Gewicht aus der ICP-Fitness. Wenn ICP zu
streng war und weniger als `registration_min_kept_chunks: 2` Chunks uebrig
bleiben, greift `enable_registration_underfill_fallback: true`: Die Pipeline
nutzt dann wieder die vorbereiteten Originalchunks, damit ein ansonsten
brauchbarer Track nicht komplett verloren geht.

## 7. Voxel-Fusion und finale Filter

Nach der Registrierung werden die Chunks voxelbasiert fusioniert. Die wichtigen
Defaults kommen aus `base.yaml`:

```yaml
fusion_voxel_size: 0.05
fusion_min_observations: 1
fusion_weight_mode: point_count
aggregate_voxel: 0.06
min_saved_aggregate_points: 50
```

Der Kern:

1. Jeder Chunk wird in 5-cm-Voxel zerlegt.
2. Punkte im selben Voxel werden pro Chunk gemittelt.
3. Gleiche Voxel aus mehreren Chunks werden zusammengefuehrt.
4. Chunk-Gewichte beeinflussen den Mittelwert.
5. Weil `fusion_min_observations: 1`, reicht schon eine Beobachtung fuer einen
   Voxel.

Danach laufen die finalen Filter:

- `enable_post_filter_stat_outlier_removal: true` entfernt statistische
  Ausreisser mit Open3D.
- `aggregate_voxel: 0.06` verdichtet die finale Punktwolke nochmals auf
  6-cm-Voxel.
- `enable_tail_bridge: false` verhindert kuenstliches Verbinden getrennter
  Komponenten.
- `symmetry_completion: true` versucht fehlende Fahrzeugseiten durch Spiegelung
  an einer geschaetzten Symmetrieebene zu ergaenzen.
- `enable_confidence_point_cap: false` laesst die Punktanzahl unangetastet.

Am Ende bekommt der Track einen Status:

- `skipped_min_hits`: zu wenige Track-Treffer
- `skipped_track_exit`: Lane-Exit-Gate nicht erfuellt
- `empty_selection`: keine brauchbaren Frames ausgewaehlt
- `empty_prepared_chunks`: nach Vorbereitung keine Chunks mehr vorhanden
- `empty_fused` oder `empty_filtered`: Fusion/Filter ergibt keine Punkte
- `skipped_min_saved_points`: weniger als `min_saved_aggregate_points`
- `skipped_quality_threshold`: Track-Qualitaet unter Threshold
- `saved`: finale Objektpunktwolke wird gespeichert

Nur `saved`-Aggregate werden als Pred-PCD/JSON ins Dataset geschrieben.

## 8. Klassifikation und GT-Matching

Wenn ein Aggregat Punkte hat, klassifiziert PointNeXt die finale Punktwolke.
Der Klassenname wird ueber `class_normalization.aliases` aus `base.yaml` auf
die TLS-Klassen normalisiert.

Danach matcht die Pipeline gespeicherte Aggregat-Tracks gegen die
MQTT-Objekthistorien. Das Matching ist one-to-one und nutzt eine Kostenmatrix
aus Track-Zentren, GT-Zentren, Extents und Zeitabstand. Die Zuordnung laeuft
ueber Hungarian Assignment.

Im Dataset-Modus werden daraus Samples in Buckets erzeugt:

- `gt-pred-same`
- `gt-pred-different`
- `unmatched_gt`
- `unmatched_pred`

## Wichtige Hinweise fuer diese Config

- `statistics_enabled: false` spart Debug- und Statusdateien, erschwert aber
  die Diagnose. Fuer Tuning-Laeufe ist `true` oft hilfreicher.
- `final_full_recompute: false` ist gut fuer Live-Betrieb, weil fertige Tracks
  inkrementell verarbeitet werden. Fuer maximale Reproduzierbarkeit kann ein
  finaler Komplettlauf nuetzlich sein.
- `warumup_runs` ist ein Tippfehler und wird von `tracking-pipeline run` nicht
  verwendet.
- `measure_runs` wird im normalen `run` ebenfalls nicht verwendet; das gehoert
  zum Benchmark-Kontext.

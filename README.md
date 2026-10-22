# Smart CCTV Exam Cheating Detection

Prototype for reviewing suspicious activity in staged exam-hall video. Alerts are intended to support a human invigilator; they are not conclusions that cheating occurred.

## Current implementation

The software work for Phases 0 through 6 is in place: the project loads `config.yaml`, reads file, RTSP, or webcam sources, runs pretrained YOLO person/phone detection and ByteTrack plus pose estimation, evaluates PEEKING and DEVICE_USE rules, saves cooldown-limited snapshots and SQLite events, and serves a live dashboard through FastAPI. The benchmark reports FPS, p95 latency, alert totals, and optional ground-truth precision/recall. Demo mode probes local clips and applies configured lower-resolution settings. Clip-based threshold tuning, the 10-minute stability run, the actual two-minute backup recording, and fresh-laptop acceptance still need to be completed with the demo videos and model weights available.

## Install

Use Python 3.10 or 3.11, then from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place recorded clips in `data/clips/` and configure their paths under `streams` in `config.yaml`. The clip files are excluded from version control. Ultralytics downloads the configured pretrained weights on first use unless they are already available locally; weights are also excluded from version control.

## Run

Check and print the parsed configuration:

```powershell
python -c "from src.config import load; print(load())"
```

Run the first configured video, write `outputs/annotated.mp4`, and save alert snapshots and events to the configured locations:

```powershell
python scripts/run_offline.py
```

Choose a configured stream and/or override its input and output:

```powershell
python scripts/run_offline.py --stream cam1 --source data/clips/hall_a.mp4 --output outputs/hall_a_annotated.mp4
```

The console reports processed-frame FPS and any generated alerts. Boxes show person track IDs and class labels; visible pose keypoints are drawn for people. Alert text is neutral and describes suspicious activity for human review.

Run the multi-stream dashboard from the project root:

```powershell
python -m src.main
```

Then open `http://127.0.0.1:8000`. The server binds to localhost so camera feeds, snapshots, and event data are not exposed on other network interfaces by default. Configure each feed in `config.yaml`; file sources loop according to their `loop` setting, while RTSP sources reconnect after interruptions.

Run the lower-load demo mode, which probes local clips and selects up to the configured `demo.max_streams`:

```powershell
python -m src.main --demo
```

The demo settings in `config.yaml` default to 480-pixel feed width and inference on every third frame. At least one readable local clip must be present. To prepare the configured browser backup, open `http://127.0.0.1:8000/static/record_backup.html`, select the dashboard tab in the browser screen-share picker, and keep it visible for `demo.recording_seconds`. Move the downloaded WebM recording into `outputs/` for the demo handoff.

Generate a benchmark report for the configured local clips:

```powershell
python scripts/benchmark.py
```

Optionally score alerts against labeled event intervals:

```powershell
python scripts/benchmark.py --ground-truth data/ground_truth.csv
```

The CSV header must be `clip,start_s,end_s,type`. The report is written to the configured `output.metrics_report` path, which defaults to `outputs/metrics_report.md`.

## Troubleshooting

- **Python is not found:** install Python 3.10/3.11 and use the Windows `py -3.11` launcher or activate the virtual environment.
- **Cannot open video:** verify the configured path from the project root, or pass `--source` with a valid file, RTSP URL, or webcam index.
- **Model download fails:** download `yolov8n.pt` and `yolov8n-pose.pt` before the demo and set their local paths under `model` in `config.yaml`.
- **CPU is slow:** set `process_every_n_frames` higher or use smaller input videos. The detector reuses the most recent detections on skipped frames.
- **No phones appear:** the pretrained COCO phone class can miss small or occluded devices; tune the configured confidence and use clear staged footage.
- **Demo mode says no readable clips:** add valid local recordings under `data/clips/` and list their paths in `config.yaml` before launching with `--demo`.
- **No metrics report is produced:** the benchmark skips missing files and needs at least one configured readable clip plus locally available pretrained weights.

## Data and privacy

Keep recordings in `data/clips/`; that directory and generated snapshots, database files, outputs, and model weights are ignored by Git. Alert snapshots intentionally save a frame for human review. The demo backup recording should also remain local in `outputs/`.

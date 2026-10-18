# Split_Step_Counter
A computer vision project that takes in a video and counts the number of times a split step has been completed.

MediaPipe pose estimation extracts body landmarks from each frame. One of
two detectors then counts split steps:

- **`lstm`** (default) is a stateful per-frame LSTM trained on labeled footage.
- **`rules`** is a hand-tuned state machine: a synchronized two-foot hop from
  a ready stance.

Use it three ways: a **desktop GUI**, an **HTTP API** (deployable to AWS),
or the **data-pipeline scripts** that build training data.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # GUI + training
pip install -r api/requirements.txt      # API (adds fastapi, uvicorn, mangum, boto3)
```

The pose model (~9 MB) downloads automatically into `models/` on first run.
The LSTM checkpoint (`models/split_step_lstm.pt`) and its normalization stats
(`models/feat_mean.npy`, `models/feat_std.npy`) are checked into git.

## GUI

```
python main.py
```

Pick a video file or start a live webcam feed. The "Use LSTM model" checkbox
switches between the two detectors.

## API

### Run locally

```
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive Swagger docs.

### Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET  | `/health` | — | Liveness check. Never requires a key. |
| POST | `/v1/count?model=lstm\|rules` | multipart, field `file` = video | Direct upload, limited by `MAX_UPLOAD_MB` (default 200). |
| POST | `/v1/upload-url` | JSON `{"filename": "match.mp4"}` | Returns a presigned S3 POST (`url`, `fields`, `bucket`, `key`) so a browser can upload directly to S3. Needs `UPLOAD_BUCKET`. |
| POST | `/v1/count/s3?model=lstm\|rules` | JSON `{"bucket": "...", "key": "..."}` | Counts a video in S3. With `UPLOAD_BUCKET` set, it only accepts keys issued by `/v1/upload-url`. |

Example:

```
curl -F "file=@match.mp4" "http://localhost:8000/v1/count?model=lstm"
```

```json
{"count": 2, "model": "rules", "events": [{"timestamp_ms": 4533.3}, {"timestamp_ms": 14800.0}],
 "frames": 463, "frames_with_pose": 456, "duration_ms": 15400.0, "fps": 30.0, "processing_s": 4.35}
```

`events` lists the video timestamp of each counted split step. From Python:

```python
import requests
with open("match.mp4", "rb") as f:
    r = requests.post("http://localhost:8000/v1/count", params={"model": "lstm"}, files={"file": f})
print(r.json()["count"])
```

Status codes:

| Code | Meaning |
|---|---|
| `400` | Empty file |
| `401` | Bad or missing key |
| `403` | S3 key not issued by this API, or not readable |
| `404` | S3 object missing |
| `413` | File too large |
| `422` | Not a readable video, or bad `model` |
| `501` | Uploads not configured (`UPLOAD_BUCKET` unset) |

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | unset | Comma-separated website origins allowed to call the API from a browser (CORS), e.g. `https://www.my-site.com`. |
| `UPLOAD_BUCKET` | unset | S3 bucket for browser uploads. Enables `/v1/upload-url` and locks `/v1/count/s3` to the keys it issues. |
| `API_KEY` | unset | If set, `/v1/*` requests must send `X-API-Key: <value>`. Use only for **server-to-server** calls, because a key embedded in website JavaScript is visible to everyone. |
| `MAX_UPLOAD_MB` | `200` | Upload size limit, for direct uploads and presigned ones. |
| `SPLIT_STEP_MODELS_DIR` | `<repo>/models` | Where the model files are read from. |

Processing runs about 3× faster than real time on a laptop CPU, so a 15 s
clip takes about 5 s.

You can count without the server too:
`python -m split_step.analyze match.mp4 --model rules`.

### Tests

```
pip install pytest httpx
python -m pytest tests -q
```

The upload tests use `assets/split_step_videos/Training Set 1/video_1.mp4`.
They are skipped when that local-only file is missing.

## Deploying to AWS for a public website

The recommended setup costs nothing while idle:

```
browser ──1. POST /v1/upload-url──────────▶ Lambda (Function URL)
        ──2. POST video (presigned)───────▶ S3 bucket, uploads/ (auto-deleted after 1 day)
        ──3. POST /v1/count/s3────────────▶ Lambda reads it from S3, returns the count
```

The video goes straight from the browser to S3, which sidesteps Lambda's
~6 MB request limit.

The public URL is protected in four ways:
- **CORS** limits browser calls to your site's origin.
- The function can **only read files uploaded through the API**.
- S3 **enforces the size limit**.
- **Reserved concurrency caps how many videos can be processed at once**, and
  so caps your bill if someone abuses the URL.

### One-command deploy

You need the [AWS CLI v2](https://aws.amazon.com/cli/) set up with
`aws configure`, and Docker Desktop running. Then, from Git Bash in the repo
root:

```bash
ALLOWED_ORIGINS="https://www.your-site.com" ./deploy/deploy_lambda.sh
```

For several origins, separate them with commas, for example
`"https://www.your-site.com,http://localhost:5500"` to also test from a
local page.

The script prints your URL at the end. Re-run it after code changes to
redeploy. It reuses everything it created the first time. Optional
variables:

| Variable | Default |
|---|---|
| `AWS_REGION` | Your configured region |
| `NAME` | `split-step-api` |
| `CONCURRENCY` | `3` |
| `MAX_UPLOAD_MB` | `200` |
| `UPLOAD_BUCKET` | `<name>-uploads-<account>-<region>` |

### Add it to your website

Copy the `<script>` from [web/example.html](web/example.html) into your
page and set `API_URL` to the printed Function URL, without a trailing slash.
To try it first, serve the file locally, e.g. `python -m http.server 5500`
in `web/`, and include `http://localhost:5500` in `ALLOWED_ORIGINS`.

### Good to know

- **Cold starts.** The first request after ~15 idle minutes takes 10–20 s
  while torch and MediaPipe load. The example page warns users about this.
- **Cost.** You pay roughly per second of processing at 3 GB of memory, a
  few tenths of a cent per short clip. S3 storage is negligible because
  uploads expire after a day. Set an
  [AWS Budget](https://console.aws.amazon.com/billing/home#/budgets) alert anyway.
- **Rate limiting.** For per-IP throttling, put API Gateway or CloudFront
  with AWS WAF in front of the function.
- **Logs.** Check CloudWatch → Log groups → `/aws/lambda/split-step-api`.

### Alternative: always-on container (ECS Fargate, EC2)

`Dockerfile` builds a plain HTTP server on port 8080 that accepts direct
uploads up to `MAX_UPLOAD_MB`. There's no S3 step, but you pay while it's
idle (roughly $30+/month for 2 vCPU / 4 GB on Fargate).

```bash
docker build --platform linux/amd64 -t split-step-api .
docker run -p 8080:8080 -e ALLOWED_ORIGINS=http://localhost:5500 split-step-api
```

Push the image to ECR and create a service from it with:
- container port **8080**
- health check path **`/health`**
- at least **2 vCPU / 4 GB**
- your `ALLOWED_ORIGINS` set

## Data pipeline (training)

```
# Record pose landmarks for one video -> CSV
python -m split_step.pose_csv_recorder VIDEO --out assets/csv_files

# Label a single recorded CSV from hand-entered ranges (csv_labeler.SPLIT_STEP_RANGES)
python -m split_step.csv_labeler assets/csv_files/pose1.csv

# Label a whole batch from pre-cut clip filenames, e.g.
#   "video_2 -00.00.06.102-00.00.06.835-seg2 (splitstep).mp4"
python -m split_step.build_labeled_dataset "assets/split_step_videos/Labeled Training Set 1" \
    "assets/split_step_videos/Training Set 1" assets/csv_files/labeled_training_set_1.csv \
    --exclude "assets/split_step_videos/Test_Set"
```

Training happens in `model_experimentation/lstm_model.ipynb`, and every run
is recorded in `model_experimentation/refinement_log.md`. After retraining,
update `_HIDDEN_SIZE` and `_THRESHOLD` in `split_step/lstm_counter.py` to
match the new checkpoint.

## Project layout

```
split_step/                  core package
  detector.py                 constants, pose helpers, rule-based SplitStepDetector, GUI worker base
  lstm_counter.py             SplitStepLSTM model + LSTMCounter (same interface as the detector)
  analyze.py                  headless "video file -> count" used by the API
  app.py                      tkinter GUI
  video_analysis.py           GUI worker for video files
  live_view_analysis.py       GUI worker for webcam input
  pose_csv_recorder.py        records pose landmarks to CSV
  csv_labeler.py              labels a single pose CSV from hand-entered ranges
  build_labeled_dataset.py    labels a batch of videos from clip filenames (+ scene-cut detection)
  paths.py                    model file locations
api/                         FastAPI app (main.py) + Lambda handler + API requirements
deploy/                      deploy_lambda.sh: one-command AWS deploy
web/                         example.html: browser client for your website
tests/                       API tests
data_windowing/              ~300 ms windowing pipeline (see TASK.md; not used by the current model)
model_experimentation/       training notebook + refinement log (not part of the app)
models/                      pose_landmarker_full.task (downloaded), split_step_lstm.pt, feat_*.npy
assets/                      recorded videos / CSVs / result plots (large media gitignored)
Dockerfile, Dockerfile.lambda  container images for AWS
main.py                      GUI entry point
```

## Known limitations

- **Recall is low.** The deployed LSTM catches about 29% of split-step frames
  on the held-out test set (see `refinement_log.md`), so counts run low.
  More diverse training data is the planned fix (`TODO.md`).
- **Camera angle matters.** Training data is mostly back-view, so side and
  front views perform worse.
- **One player at a time.** MediaPipe tracks a single pose per frame.

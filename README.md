# Split_Step_Counter
A computer vision project for tennis training. It takes a video, counts
split steps, and gives forehand, backhand and serve feedback.

MediaPipe pose estimation extracts body landmarks from each frame, and three
analyses run on them:

- **Split steps.** One of two detectors counts them:
  - **`lstm`** (default) is a stateful per-frame LSTM trained on labeled footage.
  - **`rules`** is a hand-tuned state machine: a synchronized two-foot hop
    from a ready stance.

  Each split step is scored on hop height, landing knee bend and stance width.
- **Stroke training.** Detects swings and gives feedback per stroke:
  - forehand / backhand: contact height, shoulder turn, follow-through, knee load,
    head stillness through contact, preparation timing (how early the shoulders turn)
  - serve: reach, toss height, elbow extension, trophy-position knee bend
  - **highlights**: links to your most typical, fastest and odd-one-out
    swing of each stroke
  - a **consistency score** (0–100) per stroke: how repeatable the swings
    are. Pros score about 55–70 on practice footage, because measurement
    noise counts toward the spread.
- **Recovery.** Measures how fast the player gets back to a ready stance
  (knees bent, racket in front with both hands, feet apart) after each shot.
- **Footwork.** Counts steps: steps per minute, and small adjustment steps
  between shots in a rally.
- **Fatigue trend.** On sessions of 2+ minutes, compares the first third
  with the last: hop height, knee bend, swing speed and recovery time, and
  flags the ones that dropped off.
- **Rallies.** Groups swings into rallies (shots per rally, longest rally).
- **Split step → shot timing.** Shows which shots had a split step
  0.4–2.5 s before them. Hops that happen during a swing are flagged, not
  counted as split steps.
- **Camera-angle check.** Detects whether the player is filmed from
  behind, the front or the side, and warns when the split-step model is less
  reliable for that view.
- **Coaching tips** in plain language, based on the metrics above, with a
  **coach's summary** of the top 3 things to work on, and a **practice plan**:
  a drill for each of them, with how to do it and how many reps.

Use it four ways:
- the **Tennis Trainer web app**
- an **HTTP API** (deployable to AWS)
- a **desktop GUI**
- the **data-pipeline scripts** that build training data

Any major feature can be [switched off](#feature-switches).

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

To practise strokes, pick a **Forehand / Backhand / Serve drill** and your
hitting hand. A live swing counter then runs alongside the split-step count,
which suits solo practice in front of a webcam. Live swing detection is
approximate: on test clips it matched the offline analysis on about 85% of
swings, usually one frame later, and it can miss the first swing. For full
metrics, use the web app or API.

With **Sound cues** ticked, the GUI beeps on every split step (high tone)
and swing (lower tone), so you can drill alone without watching the screen.

## API and web app

### Run locally

```
uvicorn api.main:app --reload --port 8000
```

| URL | What it is |
|---|---|
| http://localhost:8000/app/ | **Tennis Trainer** web app: pick split steps, forehand, backhand, serve or match play (auto) and upload (or drag in) a clip. Click any timestamp or timeline marker to jump the video there. Past sessions appear in a progress table. |
| http://localhost:8000/docs | Interactive Swagger docs |

### Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET  | `/health` | — | Liveness check. Never requires a key. |
| GET  | `/v1/features` | — | Which [features](#feature-switches) are on, so UIs can hide the rest. |
| POST | `/v1/count` | multipart, field `file` = video | Direct upload, limited by `MAX_UPLOAD_MB` (default 200). |
| POST | `/v1/upload-url` | JSON `{"filename": "match.mp4"}` | Returns a presigned S3 POST (`url`, `fields`, `bucket`, `key`) so a browser can upload directly to S3. Needs `UPLOAD_BUCKET`. |
| POST | `/v1/report` | JSON `{"result": <count response>, "title": "match.mp4"}` | Returns the result as a readable Markdown report (`text/markdown`), the same as `analyze --report`. Nothing is stored. |
| POST | `/v1/count/s3` | JSON `{"bucket": "...", "key": "..."}` | Analyses a video in S3. With `UPLOAD_BUCKET` set, it only accepts keys issued by `/v1/upload-url`. |

Both analysis endpoints take these query parameters:

| Parameter | Values | Meaning |
|---|---|---|
| `model` | `lstm` (default), `rules` | Split-step detector |
| `require_full_pose` | `true` (default), `false` | Skip the LSTM on frames where the hips, ankles, heels or toes aren't visible. The LSTM was only trained on fully visible frames. |
| `drill` | omit, `forehand`, `backhand`, `serve`, `auto` | Stroke training. In a drill, every swing is scored as that stroke. `auto` classifies each swing; serve detection is reliable, but forehand vs backhand is experimental. |
| `hand` | `auto` (default), `right`, `left` | Hitting hand for stroke training |
| `include_pose` | `false` (default), `true` | Also return the per-frame skeleton for drawing overlays: ~7 KB per second, capped at ~1 MB |

Example:

```
curl -F "file=@practice.mp4" "http://localhost:8000/v1/count?model=rules&drill=forehand"
```

Response, shortened:

```json
{"count": 8, "model": "rules",
 "events": [{"timestamp_ms": 4500.0, "hop_height": 0.206, "landing_knee_angle": 101.4, "landing_stance_ratio": 1.09}],
 "session": {"split_steps_per_minute": 25.22, "interval_s": {"median": 1.233},
             "tips": ["Solid split steps: good knee bend, base width and rhythm."]},
 "strokes": {"drill": "forehand", "hand": "right",
   "swings": [{"timestamp_ms": 733.3, "stroke": "forehand", "peak_wrist_speed": 9.1, "contact_height": 0.42,
               "shoulder_turn_deg": 124.1, "finish_above_shoulder_m": 0.22, "load_knee_angle": 115.6}],
   "summary": {"forehand": {"count": 8, "contact_height": {"median": 0.42}}},
   "tips": ["Forehand: technique metrics look solid. Keep it up!"]},
 "frames_tracked": 572, "frames": 572, "frames_with_pose": 572, "duration_ms": 19033.3, "fps": 30.0}
```

What the metrics mean:

| Metric | Unit / scale |
|---|---|
| `hop_height` | Fraction of the player's torso length |
| `landing_stance_ratio` | Ankle spread ÷ shoulder width |
| Angles | Degrees; 180 = straight |
| `contact_height` | 0 = hips, 1 = shoulders |
| `head_movement_m` | Metres the head (nose) moves around contact; pros measured ~0.06 m |
| `prep_lead_ms` | How long before contact the shoulders were turned. Pros measured medians of 0.3–0.5 s on forehands (the coil continues until just before the swing) and 0.7–1.3 s on backhands. Capped at ~1.4 s (turned before the measuring window) |
| `peak_wrist_speed` | m/s of the hand, not the racket head. Useful for comparing your own swings, not as an absolute speed. |

The coaching-tip thresholds are heuristics, not values fitted to data.
Tune them in `split_step/tennis_metrics.py` and `split_step/strokes.py`.

From Python:

```python
import requests
with open("practice.mp4", "rb") as f:
    r = requests.post("http://localhost:8000/v1/count", params={"drill": "serve"}, files={"file": f})
print(r.json()["strokes"]["tips"])
```

Status codes:

| Code | Meaning |
|---|---|
| `400` | Empty file, or nothing to analyse |
| `401` | Bad or missing key |
| `403` | S3 key not issued by this API, or not readable |
| `404` | S3 object missing, or the feature is switched off |
| `413` | File too large, or video longer than `MAX_VIDEO_SECONDS` |
| `429` | Rate limit reached (`RATE_LIMIT_PER_HOUR`) |
| `422` | Not a readable video, or a bad parameter |
| `501` | Uploads not configured (`UPLOAD_BUCKET` unset) |

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | unset | Comma-separated website origins allowed to call the API from a browser (CORS), e.g. `https://www.my-site.com`. Not needed for the built-in `/app/`. |
| `UPLOAD_BUCKET` | unset | S3 bucket for browser uploads. Enables `/v1/upload-url` and locks `/v1/count/s3` to the keys it issues. Without it, the web app uploads directly. |
| `API_KEY` | unset | If set, `/v1/*` requests must send `X-API-Key: <value>`. Use only for **server-to-server** calls, because a key embedded in website JavaScript is visible to everyone. |
| `MAX_UPLOAD_MB` | `200` | Upload size limit, for direct uploads and presigned ones. |
| `MAX_VIDEO_SECONDS` | `600` | Longest video analysed; longer ones get `413`. Keeps each request well inside Lambda's 15-minute limit. |
| `RATE_LIMIT_PER_HOUR` | `0` (off) | Analyses per client IP per hour; beyond that the API returns `429` with `Retry-After`. Per container, so the effective limit scales with reserved concurrency. |
| `SPLIT_STEP_FEATURES` | unset | Feature switches, see below. |
| `SPLIT_STEP_MODELS_DIR` | `<repo>/models` | Where the model files are read from. |

Processing runs about 3× faster than real time on a laptop CPU, so a 15 s
clip takes about 5 s.

Videos above 30 fps (common on phones) are analysed at ~30 fps by skipping
frames. That matches the training data and halves the cost. On a 60 fps
match clip, analysing every frame added 5 swings, at least 3 of them false.

You can analyse without the server too:
`python -m split_step.analyze practice.mp4 --model rules --drill serve`.
Add `--report session.md` for a readable Markdown report (coach's summary,
practice plan, tables of split steps and swings) to share with a coach, and
`--annotate annotated.mp4` for a copy of the video with the skeleton, a
"SPLIT STEP" banner, each stroke with its wrist speed, and running counts
drawn on.

To analyse a whole folder of clips into one spreadsheet (one row per clip:
split steps, consistency, recovery, camera view, top tip, ...):

```
python -m split_step.batch "my clips" --drill auto --out summary.csv --json-dir reports --report-dir reports
```

A clip that can't be read gets a row with its error; the rest still run.

### Feature switches

Every major feature has an on/off switch. A switched-off feature disappears
everywhere:
- The **web app** removes its controls (it reads `/v1/features`).
- The **desktop GUI** hides its widgets.
- The **API** leaves it out of responses and returns `404` for requests that ask for it.

| Switch | Controls |
|---|---|
| `split_step_counting` | Split-step counting (parent of `split_step_metrics`) |
| `split_step_metrics` | Per-split-step quality metrics and session summary |
| `coach_summary` | Prioritised summary: the top 3 things to work on, plus strengths |
| `practice_plan` | Drills for the next session, one per focus point of the coach's summary (needs `coach_summary`) |
| `footwork` | Step counting: steps per minute, adjustment steps between shots |
| `fatigue_trend` | Compare the start and end of sessions of 2+ minutes: does the player fade? |
| `camera_view_check` | Detect the camera angle; warn when it's one the model handles poorly |
| `stroke_training` | All stroke training (parent of the four below) |
| `forehand_training`, `backhand_training`, `serve_training` | Each drill |
| `auto_stroke_detection` | "Match play (auto)" mode |
| `split_step_timing` | "Split step before shot": links each swing to the split step before it (needs counting and stroke training) |
| `lstm_model`, `rules_model` | Each split-step model |
| `direct_upload`, `browser_uploads` | API upload routes |
| `live_view` | Webcam button in the desktop GUI |
| `audio_cues` | Desktop GUI: a high beep for each split step, a lower one for each swing (with a checkbox to mute) |
| `session_history` | Web app: progress table and chart of past sessions, saved in the browser |
| `history_transfer` | Web app: export your history to a JSON file and import it in another browser or device (merged, no duplicates) |
| `session_compare` | Web app: tick two past sessions to compare every metric side by side, marked better / worse (needs `session_history`) |
| `drill_goals` | Web app: set targets (split-step rate, consistency, recovery, ...) and see which ones each session met |
| `swing_chart` | Web app: each swing's wrist speed through the clip, coloured by stroke; click a dot to watch it |
| `swing_highlights` | Most typical, fastest and odd-one-out swing of each stroke (with the metric that made it odd), linked to the video |
| `rally_stats` | Group swings into rallies: shots per rally, longest rally |
| `ready_position` | Recovery to a ready stance after each shot |
| `playback_controls` | Web app: slow motion (½×, ¼×), previous/next shot buttons, and keys `[` `]` (shot to shot) and `,` `.` (one frame) |
| `markdown_report` | `POST /v1/report` and the web app's "Download report (Markdown)" button |
| `report_download` | Web app: download results as JSON or CSV, or print / save a clean PDF report |
| `pose_overlay` | Web app: skeleton drawn over the video (API: `include_pose`) |

Everything is on by default. You can override switches in two places, and
the environment variable wins:

```bash
# 1. features.json in the repo root (or a path in SPLIT_STEP_FEATURES_FILE)
{"serve_training": false, "auto_stroke_detection": false}

# 2. Environment variable
SPLIT_STEP_FEATURES="serve_training=off,live_view=off" uvicorn api.main:app
```

Switches are re-read on every request. On Lambda, set the variable with
`aws lambda update-function-configuration`. Merge it with the existing
variables, because that command replaces them all.

### Tests

```
pip install pytest httpx
python -m pytest tests -q            # Python: API, metrics, strokes, feature switches
cd tests/web && npm install && npm test  # web app logic, in jsdom
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
- **Per-IP rate limits** (20 analyses per hour by default) and a **10-minute
  video cap**.

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
| `MAX_VIDEO_SECONDS` | `600` |
| `RATE_LIMIT_PER_HOUR` | `20` |
| `UPLOAD_BUCKET` | `<name>-uploads-<account>-<region>` |

### Add it to your website

Copy the `<script>` from [web/index.html](web/index.html) into your
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
  report.py                   a result as a Markdown report
  annotate.py                 the analysis drawn onto a copy of the video
  batch.py                    analyse a folder of clips into one summary CSV
  analyze.py                  headless "video file -> results" used by the API
  pose_track.py               whole-video landmarks as numpy arrays
  tennis_metrics.py           split-step quality metrics, session summary, tips
  strokes.py                  swing detection + forehand/backhand/serve training
  timing.py                   links split steps to the shots that follow
  rallies.py                  groups swings into rallies
  ready_position.py           recovery to a ready stance after each shot
  footwork.py                 step counting, adjustment steps between shots
  highlights.py               typical / fastest / odd-one-out swing per stroke
  fatigue.py                  early vs late session comparison
  camera_view.py              camera-angle detection
  coach.py                    prioritised coach's summary of all tips
  practice.py                 drills for the coach's focus points
  features.py                 feature switches
  app.py                      tkinter GUI
  video_analysis.py           GUI worker for video files
  live_view_analysis.py       GUI worker for webcam input
  pose_csv_recorder.py        records pose landmarks to CSV
  csv_labeler.py              labels a single pose CSV from hand-entered ranges
  build_labeled_dataset.py    labels a batch of videos from clip filenames (+ scene-cut detection)
  paths.py                    model file locations
api/                         FastAPI app (main.py) + Lambda handler + API requirements
deploy/                      deploy_lambda.sh: one-command AWS deploy
web/                         index.html: Tennis Trainer web app (served by the API at /app/)
tests/                       pytest suite; tests/web: jsdom tests for the web app
data_windowing/              ~300 ms windowing pipeline (see TASK.md; not used by the current model)
model_experimentation/       training notebook + refinement log (not part of the app)
models/                      pose_landmarker_full.task (downloaded), split_step_lstm.pt, feat_*.npy
assets/                      recorded videos / CSVs / result plots (large media gitignored)
Dockerfile, Dockerfile.lambda  container images for AWS
main.py                      GUI entry point
```

## Known limitations

- **Split-step timing inherits the counter's low recall.** "Only N of M
  shots had a split step" undercounts whenever the detector misses a split
  step.
- **Recall is low.** The deployed LSTM catches about 29% of split-step frames
  on the held-out test set (see `refinement_log.md`), so counts run low.
  More diverse training data is the planned fix (`TODO.md`).
- **Camera angle matters.** Training data is mostly back-view, so side and
  front views perform worse.
- **One player at a time.** MediaPipe tracks a single pose per frame.

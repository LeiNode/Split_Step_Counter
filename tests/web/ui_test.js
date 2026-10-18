// Drives web/index.html in jsdom with a mocked API (fixture = a real analysis
// of a pro practice clip). Run:  cd tests/web && npm install && npm test
const fs = require("fs");
const { JSDOM } = require("jsdom");
const path = require("path");
const page = path.join(__dirname, "..", "..", "web", "index.html");
const respPath = path.join(__dirname, "analysis_fixture.json");
const html = fs.readFileSync(page, "utf8");
const analysis = JSON.parse(fs.readFileSync(respPath, "utf8"));
let failures = 0;
const check = (cond, msg) => { console.log((cond ? "ok   " : "FAIL ") + msg); if (!cond) failures++; };
const tick = () => new Promise((r) => setTimeout(r, 30));

async function load(flags, { uploadsConfigured = false } = {}) {
  const calls = [];
  const dom = new JSDOM(html, {
    url: "http://localhost:8000/app/", runScripts: "dangerously", pretendToBeVisual: true,
    beforeParse(window) {
      window.URL.createObjectURL = () => "blob:fake";
      window.HTMLMediaElement.prototype.play = () => {};
      window.fetch = async (url, opts = {}) => {
        calls.push(url);
        const json = (status, body) => ({ ok: status < 400, status, json: async () => body });
        if (url.endsWith("/v1/features")) return json(200, { features: flags });
        if (url.includes("/v1/upload-url"))
          return uploadsConfigured ? json(500, { detail: "boom" }) : json(501, { detail: "Browser uploads are not configured (UPLOAD_BUCKET unset)." });
        if (url.includes("/v1/count")) return json(200, analysis);
        return json(404, {});
      };
    },
  });
  await tick();
  return { dom, doc: dom.window.document, calls };
}

const ALL = { drill_goals: true, pose_overlay: true, report_download: true, rally_stats: true, session_history: true, split_step_timing: true, split_step_counting: true, split_step_metrics: true, stroke_training: true,
  forehand_training: true, backhand_training: true, serve_training: true, auto_stroke_detection: true,
  lstm_model: true, rules_model: true, direct_upload: true, browser_uploads: true, live_view: true };

function chooseFile(dom) {
  const input = dom.window.document.getElementById("video");
  const file = new dom.window.File(["x"], "clip.mp4", { type: "video/mp4" });
  Object.defineProperty(input, "files", { value: [file] });
  input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
}

(async () => {
  // 1. Everything on
  let { dom, doc, calls } = await load(ALL);
  const modes = () => [...doc.querySelectorAll('input[name="mode"]')].map((i) => i.value);
  check(modes().join() === "split,forehand,backhand,serve,auto", "all five modes shown: " + modes());
  check(doc.getElementById("go").disabled, "Analyse disabled before a file is chosen");
  chooseFile(dom);
  check(!doc.getElementById("go").disabled, "Analyse enabled after choosing a file");
  // Goals: one achievable (the fixture has 8 swings in one rally), one not, one unmeasurable
  const setGoal = (key, v) => { const i = doc.getElementById("goal-" + key); i.value = v; i.dispatchEvent(new dom.window.Event("change")); };
  check(doc.querySelectorAll("#goals-form input").length === 6, "goal inputs built");
  setGoal("rally", 5); setGoal("with_split", 90); setGoal("split_rate", 1);
  doc.querySelector('input[value="auto"]').click();
  doc.getElementById("model").value = "rules";
  doc.getElementById("go").click();
  await tick(); await tick();
  const countCall = calls.find((u) => u.includes("/v1/count?"));
  check(countCall.includes("include_pose=true"), "asks for the skeleton when the overlay is on");
  check(countCall && countCall.includes("drill=auto") && countCall.includes("model=rules") && countCall.includes("hand=auto"),
        "falls back to direct upload with drill/model/hand: " + countCall);
  const goalText = doc.getElementById("goal-list").textContent;
  check(!doc.getElementById("goal-results").hidden && doc.querySelectorAll("#goal-list li").length === 3, "three goals listed");
  check(/✓ Met · Longest rally/.test(goalText), "achieved goal marked met: " + goalText.slice(0, 60));
  check(/✗ Not yet · Shots with a split step before them/.test(goalText), "missed goal marked not yet");
  check(/✓ Met · Split steps per minute/.test(goalText), "split-step goal met");
  setGoal("rally", ""); setGoal("with_split", ""); setGoal("split_rate", "");
  check(Object.keys(JSON.parse(dom.window.localStorage.getItem("tennis-trainer-goals-v1"))).length === 0, "clearing a box removes the goal");
  check(!doc.getElementById("split-results").hidden, "split-step results shown");
  check(!doc.getElementById("camera-notice").hidden && doc.getElementById("camera-notice").textContent.includes("film from behind"),
        "camera-angle warning shown");
  check(doc.querySelectorAll("#split-table tr").length === analysis.events.length + 1, "one split-step row per event");
  check(!doc.getElementById("stroke-results").hidden, "stroke results shown");
  check(doc.querySelectorAll("#stroke-table tr").length === analysis.strokes.swings.length + 1, "one row per swing");
  check(doc.getElementById("stroke-table").textContent.includes("backhand"), "stroke types rendered");
  check(doc.getElementById("stroke-tips").children.length > 0, "stroke tips rendered");
  check(doc.getElementById("stroke-stats").textContent.includes("Forehand consistency"), "consistency score shown");
  check(doc.getElementById("stroke-stats").textContent.includes("Split step before shot"), "timing stat shown");
  check(doc.getElementById("stroke-table").textContent.includes("Split step before"), "lead-time column shown");
  check(doc.getElementById("split-table").textContent.includes("hop during a swing"), "swing hops marked");
  const link = doc.querySelector("#split-table a.ts");
  link.click();
  check(Math.abs(doc.getElementById("player").currentTime - Math.max(0, analysis.events[0].timestamp_ms / 1000 - 1)) < 0.01,
        "clicking a timestamp seeks the video");
  check(doc.getElementById("status").textContent.startsWith("Done"), "status says done: " + doc.getElementById("status").textContent);

  // Timeline: one marker per real split step + one per swing
  const expected = analysis.events.filter((e) => !e.during_swing).length + analysis.strokes.swings.length;
  const markers = doc.querySelectorAll("#timeline .mk");
  check(!doc.getElementById("timeline-box").hidden && markers.length === expected,
        `timeline has ${markers.length} markers (expected ${expected})`);
  check(doc.querySelectorAll("#timeline .mk.k-backhand").length === analysis.strokes.swings.filter((s) => s.stroke === "backhand").length,
        "markers coloured by stroke");
  const last = markers[markers.length - 1];
  last.click();
  check(doc.getElementById("player").currentTime > 0, "clicking a marker seeks the video");

  // Rallies
  check(doc.getElementById("stroke-stats").textContent.includes("Longest rally"), "rally stats shown");
  check(doc.getElementById("stroke-table").textContent.includes("Rally"), "rally column shown");
  check(doc.getElementById("stroke-table").textContent.includes("Back to ready"), "recovery column shown");
  check(doc.getElementById("stroke-stats").textContent.includes("Recovery"), "recovery stat shown");

  // Downloads: capture the generated CSV
  let saved = null;
  dom.window.URL.createObjectURL = (blob) => { saved = blob; return "blob:x"; };
  check(!doc.getElementById("downloads").hidden, "download buttons shown");
  doc.getElementById("dl-csv").click();
  const readBlob = (b) => new Promise((r) => { const fr = new dom.window.FileReader(); fr.onload = () => r(fr.result); fr.readAsText(b); });
  const csv = saved ? await readBlob(saved) : "";
  const lines = csv.trim().split("\n");
  check(lines.length === 1 + analysis.events.length + analysis.strokes.swings.length, `CSV has a row per event and swing (${lines.length - 1})`);
  check(lines[0].startsWith("type,timestamp_ms"), "CSV header: " + lines[0].slice(0, 40));
  check(csv.includes("forehand") && csv.includes("split step"), "CSV contains shots and split steps");
  doc.getElementById("dl-json").click();
  check(saved && JSON.parse(await readBlob(saved)).count === analysis.count, "JSON report round-trips");

  // History: saved after each analysis, clearable
  const rows = () => doc.querySelectorAll("#history-table tr").length - 1;
  check(!doc.getElementById("history").hidden && rows() === 1, "history has the session");
  check(doc.getElementById("history-table").textContent.includes("Match play"), "history shows the mode");
  doc.getElementById("go").click();
  await tick(); await tick();
  check(rows() === 2, "second analysis adds a row");
  // Progress chart: appears from the second session, one marker per session
  check(!doc.getElementById("chart-wrap").hidden, "progress chart shown with two sessions");
  const metricOpts = [...doc.querySelectorAll("#chart-metric option")].map((o) => o.value);
  check(metricOpts.includes("per_min") && metricOpts.includes("consistency"), "chart metrics offered: " + metricOpts);
  check(doc.querySelectorAll("#chart circle.pt").length === 2 && doc.querySelector("#chart polyline"), "line with a marker per session");
  doc.querySelector("#chart circle.pt").dispatchEvent(new dom.window.FocusEvent("focus"));
  check(!doc.getElementById("chart-tip").hidden && doc.getElementById("chart-tip").textContent.includes("clip.mp4"),
        "focusing a point shows its tooltip");
  const sel = doc.getElementById("chart-metric");
  sel.value = "knee"; sel.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  check(doc.getElementById("chart").getAttribute("aria-label").startsWith("Landing knee"), "switching metric redraws");
  const svgOut = doc.getElementById("chart").outerHTML;
  doc.getElementById("clear-history").click();
  check(doc.getElementById("history").hidden, "clear history hides the section");
  if (process.env.CHART_SVG_OUT) require("fs").writeFileSync(process.env.CHART_SVG_OUT, svgOut);

  // 2. Serve + auto off, rules model off
  ({ dom, doc } = await load({ ...ALL, serve_training: false, auto_stroke_detection: false, rules_model: false }));
  check(modes().join() === "split,forehand,backhand", "disabled modes removed: " + modes());
  ({ dom, doc, calls } = await load({ ...ALL, session_history: false, report_download: false, pose_overlay: false, drill_goals: false }));
  check(!doc.getElementById("goal-settings") && !doc.getElementById("goal-results"), "goals removed when switched off");
  check(!doc.getElementById("history"), "history section removed when switched off");
  check(!doc.getElementById("overlay") && !doc.getElementById("show-skeleton"), "overlay removed when switched off");
  chooseFile(dom); doc.getElementById("go").click(); await tick(); await tick();
  check(!calls.some((u) => u.includes("include_pose")), "no skeleton requested when overlay is off");
  check(!doc.getElementById("downloads"), "download buttons removed when switched off");
  ({ dom, doc } = await load({ ...ALL, serve_training: false, auto_stroke_detection: false, rules_model: false }));
  check([...doc.querySelectorAll("#model option")].map((o) => o.value).join() === "lstm", "disabled model removed");

  // 3. Stroke training off entirely
  ({ dom, doc } = await load({ ...ALL, stroke_training: false, forehand_training: false, backhand_training: false,
                               serve_training: false, auto_stroke_detection: false }));
  check(modes().join() === "split", "only split steps left");
  check(!doc.getElementById("hand"), "hand picker removed with stroke training");

  // 4. Split steps off: first stroke mode becomes default, model picker hidden
  ({ dom, doc } = await load({ ...ALL, split_step_counting: false, split_step_metrics: false }));
  check(doc.querySelector('input[name="mode"]:checked').value === "forehand", "first remaining mode preselected");
  check(doc.getElementById("model-picker").hidden, "model picker hidden without split-step counting");

  // 5. Upload error surfaces (not silently swallowed) when uploads ARE configured
  ({ dom, doc } = await load(ALL, { uploadsConfigured: true }));
  chooseFile(dom);
  doc.getElementById("go").click();
  await tick(); await tick();
  check(doc.getElementById("status").textContent.includes("boom"), "real upload errors shown: " + doc.getElementById("status").textContent);

  console.log(failures ? `${failures} FAILED` : "ALL PASSED");
  process.exit(failures ? 1 : 0);
})();

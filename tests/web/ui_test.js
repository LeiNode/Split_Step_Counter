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

const ALL = { session_history: true, split_step_timing: true, split_step_counting: true, split_step_metrics: true, stroke_training: true,
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
  doc.querySelector('input[value="auto"]').click();
  doc.getElementById("model").value = "rules";
  doc.getElementById("go").click();
  await tick(); await tick();
  const countCall = calls.find((u) => u.includes("/v1/count?"));
  check(countCall && countCall.includes("drill=auto") && countCall.includes("model=rules") && countCall.includes("hand=auto"),
        "falls back to direct upload with drill/model/hand: " + countCall);
  check(!doc.getElementById("split-results").hidden, "split-step results shown");
  check(doc.querySelectorAll("#split-table tr").length === analysis.events.length + 1, "one split-step row per event");
  check(!doc.getElementById("stroke-results").hidden, "stroke results shown");
  check(doc.querySelectorAll("#stroke-table tr").length === analysis.strokes.swings.length + 1, "one row per swing");
  check(doc.getElementById("stroke-table").textContent.includes("backhand"), "stroke types rendered");
  check(doc.getElementById("stroke-tips").children.length > 0, "stroke tips rendered");
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

  // History: saved after each analysis, clearable
  const rows = () => doc.querySelectorAll("#history-table tr").length - 1;
  check(!doc.getElementById("history").hidden && rows() === 1, "history has the session");
  check(doc.getElementById("history-table").textContent.includes("Match play"), "history shows the mode");
  doc.getElementById("go").click();
  await tick(); await tick();
  check(rows() === 2, "second analysis adds a row");
  doc.getElementById("clear-history").click();
  check(doc.getElementById("history").hidden, "clear history hides the section");

  // 2. Serve + auto off, rules model off
  ({ dom, doc } = await load({ ...ALL, serve_training: false, auto_stroke_detection: false, rules_model: false }));
  check(modes().join() === "split,forehand,backhand", "disabled modes removed: " + modes());
  ({ dom, doc } = await load({ ...ALL, session_history: false }));
  check(!doc.getElementById("history"), "history section removed when switched off");
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

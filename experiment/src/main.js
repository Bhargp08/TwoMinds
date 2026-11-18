import { initJsPsych } from "jspsych";
import "jspsych/css/jspsych.css";
import htmlButtonResponse from "@jspsych/plugin-html-button-response";
import fullscreen from "@jspsych/plugin-fullscreen";

import { N_TRIALS_PER_BLOCK, loadConfig } from "./lib/config.js";
import { blockOrderForCode } from "./lib/counterbalance.js";
import { generateParticipantCode, isValidParticipantCode } from "./lib/participantCode.js";
import { pipeFilename } from "./lib/pipeExtension.js";
import { createSaver, trialsToCsv } from "./lib/save.js";
import { buildTaskTimeline, collectTrialRows } from "./lib/task.js";
import "./style.css";

const CONSENT_HTML = `
<div class="prose">
  <h1>TwoMinds</h1>
  <p>This is a <strong>student project</strong> with volunteer participants.
  It is <strong>not IRB-approved research</strong>.</p>
  <p>Participation is <strong>voluntary</strong>. You may quit at any time.
  Data are <strong>anonymous</strong>: we store a random code only — no names,
  emails, or IP addresses.</p>
  <p>You can ask for your data to be deleted by giving your participant code.</p>
  <p>Each session has a short practice, a brief quiz, then two blocks of
  the spaceship / planet game (one with time pressure, one without).</p>
</div>`;

function showError(message) {
  document.body.innerHTML = `<p class="error">${message}</p>`;
}

function downloadText(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

let config;
try {
  config = loadConfig({
    mode: import.meta.env.VITE_TWOMINDS_MODE,
    experimentId: import.meta.env.VITE_DATAPIPE_EXPERIMENT_ID,
  });
} catch (e) {
  showError(`Configuration error: ${e.message}`);
  throw e;
}

const participantCode = generateParticipantCode();
if (!isValidParticipantCode(participantCode)) {
  showError("Could not assign a valid participant code.");
  throw new Error("invalid participant code");
}
const blockOrder = blockOrderForCode(participantCode);

const params = new URLSearchParams(window.location.search);
const SIMULATE = params.has("simulate");
const QUICK = params.has("quick");
const nTrials = QUICK ? 2 : N_TRIALS_PER_BLOCK;

let saver;
try {
  saver = createSaver({
    mode: config.mode,
    experimentId: config.experimentId,
    simulate: SIMULATE,
    getFilename: () => pipeFilename({ participantCode }),
    onSave: (result) => {
      if (!result.ok) {
        showError(
          "Your data could not be saved. Please contact the researcher and keep this page open.",
        );
      }
    },
  });
} catch (e) {
  showError(`This build cannot save data yet: ${e.message}`);
  throw e;
}

const jsPsych = initJsPsych({
  extensions: saver.extensions ?? [],
  on_finish: finish,
});

jsPsych.data.addProperties({
  participant_code: participantCode,
  block_order: blockOrder.join(","),
});

async function finish() {
  if (SIMULATE && config.mode !== "mock") {
    showError("Simulated session complete (DataPipe disabled; nothing was saved).");
    return;
  }
  if (config.mode !== "mock") return;

  const rows = collectTrialRows(jsPsych);
  const csv = trialsToCsv(rows);
  const filename = pipeFilename({ participantCode });
  await saver.save(filename, csv);
  downloadText(filename, csv);
  document.body.innerHTML = `<p class="done">Session complete (mock mode: a CSV was downloaded; nothing was sent anywhere).
    Your code is <code>${participantCode}</code>.</p>`;
}

const timeline = [
  {
    type: fullscreen,
    fullscreen_mode: true,
    message: "<p>This task runs in full screen. Click to continue.</p>",
  },
  {
    type: htmlButtonResponse,
    stimulus: CONSENT_HTML,
    choices: ["I agree, continue"],
  },
  {
    type: htmlButtonResponse,
    stimulus: `<div class="prose">
      <p>Your anonymous participant code is
      <strong style="font-family:monospace;font-size:1.5rem">${participantCode}</strong>.</p>
      <p>Note it down if you may later ask for deletion by code.</p>
      <p>Your block order: ${blockOrder.map((c) => c.replaceAll("_", " ")).join(" → ")}.</p>
    </div>`,
    choices: ["Start"],
  },
  ...buildTaskTimeline({
    participantCode,
    blockOrder,
    nTrials,
    practiceTrials: QUICK ? 1 : 4,
  }),
  {
    type: htmlButtonResponse,
    stimulus: `<div class="prose"><h2>Thanks</h2>
      <p>That is the end of the session. Remember your code
      (<strong style="font-family:monospace">${participantCode}</strong>)
      if you want your data deleted later.</p></div>`,
    choices: ["Finish"],
  },
];

jsPsych.run(timeline);

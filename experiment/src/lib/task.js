import htmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";
import htmlButtonResponse from "@jspsych/plugin-html-button-response";
import { deadlinesForCondition } from "./config.js";
import { getRewardSchedule } from "./schedule.js";
import { scheduleSeedForCondition } from "./session.js";

const P_COMMON = 0.7;

const PLANETS = {
  1: { name: "red planet", color: "#c45c4a", aliens: ["Zorp", "Blip"] },
  2: { name: "blue planet", color: "#3a6ea5", aliens: ["Kliq", "Norf"] },
};

function shipHtml(side) {
  const label = side === 0 ? "Ship A (F)" : "Ship B (J)";
  return `<div class="choice-card">${label}</div>`;
}

function planetHtml(s2) {
  const p = PLANETS[s2];
  return `<div class="planet" style="background:${p.color}">${p.name}</div>`;
}

function alienHtml(s2, a2) {
  const name = PLANETS[s2].aliens[a2];
  return `<div class="choice-card">Ask ${name} ( ${a2 === 0 ? "F" : "J"} )</div>`;
}

/**
 * Build practice + quiz + two experimental blocks for the two-step task.
 */
export function buildTaskTimeline({
  participantCode,
  blockOrder,
  nTrials,
  practiceTrials = 4,
}) {
  const timeline = [];

  timeline.push({
    type: htmlButtonResponse,
    stimulus: `
      <div class="prose">
        <h2>How to play</h2>
        <p>Each trial has two stages. First you pick a spaceship (F or J).
        Each ship usually goes to one planet, but sometimes takes a rare route.</p>
        <p>On the planet you ask one of two aliens for treasure. Some aliens are
        better than others, and how good they are drifts slowly over time.</p>
        <p>Press F for the left option, J for the right. If you are too slow,
        the trial is missed.</p>
      </div>`,
    choices: ["Start practice"],
  });

  timeline.push(
    ...buildBlockTrials({
      participantCode,
      block: -1,
      condition: "no_pressure",
      nTrials: practiceTrials,
      scheduleSeed: 0,
      isPractice: true,
    }),
  );

  timeline.push(buildComprehensionQuiz());

  blockOrder.forEach((condition, block) => {
    const label =
      condition === "no_pressure"
        ? "No time pressure — take your time (but still respond before the deadline)."
        : "Time pressure — respond quickly!";
    timeline.push({
      type: htmlButtonResponse,
      stimulus: `<div class="prose"><h2>Block ${block + 1} of ${blockOrder.length}</h2><p>${label}</p></div>`,
      choices: ["Begin"],
    });
    timeline.push(
      ...buildBlockTrials({
        participantCode,
        block,
        condition,
        nTrials,
        scheduleSeed: scheduleSeedForCondition(condition),
        isPractice: false,
      }),
    );
  });

  return timeline;
}

function buildComprehensionQuiz(_jsPsych) {
  const questions = [
    {
      prompt: "Each spaceship usually goes to one planet, but not always.",
      correct: 0,
      choices: ["True", "False"],
    },
    {
      prompt: "How good each alien is at giving treasure stays fixed forever.",
      correct: 1,
      choices: ["True", "False"],
    },
    {
      prompt: "If you miss the deadline, what happens?",
      correct: 0,
      choices: ['You see "Too slow" and that trial is missed', "You get treasure anyway"],
    },
  ];

  return {
    timeline: [
      {
        type: htmlButtonResponse,
        stimulus: `<div class="prose"><p>Quick check — please answer correctly to continue.</p></div>`,
        choices: ["OK"],
      },
      ...questions.map((q) => ({
        type: htmlButtonResponse,
        stimulus: `<div class="prose"><p>${q.prompt}</p></div>`,
        choices: q.choices,
        data: { trial_kind: "quiz", correct_index: q.correct },
      })),
    ],
    loop_function(data) {
      const answers = data.values().filter((d) => d.trial_kind === "quiz").slice(-questions.length);
      return !answers.every((d, i) => d.response === questions[i].correct);
    },
  };
}

function buildBlockTrials({
  participantCode,
  block,
  condition,
  nTrials,
  scheduleSeed,
  isPractice,
}) {
  const schedule = getRewardSchedule(scheduleSeed);
  const deadlines = deadlinesForCondition(condition);
  const trials = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const trialState = {
      a1: null,
      s2: null,
      a2: null,
      r: null,
      transition: null,
      rt1: null,
      rt2: null,
      missed_a1: false,
      missed_a2: false,
    };

    trials.push({
      type: htmlKeyboardResponse,
      stimulus: `<div class="stage">
        <p>Choose a spaceship</p>
        <div class="row">${shipHtml(0)}${shipHtml(1)}</div>
      </div>`,
      choices: ["f", "j"],
      trial_duration: deadlines.stage1,
      response_ends_trial: true,
      data: {
        trial_kind: isPractice ? "practice_stage1" : "stage1",
        participant_code: participantCode,
        block,
        condition,
        trial,
        schedule_seed: scheduleSeed,
      },
      on_finish(data) {
        if (data.response == null) {
          trialState.missed_a1 = true;
          trialState.rt1 = deadlines.stage1;
        } else {
          trialState.a1 = data.response === "f" ? 0 : 1;
          trialState.rt1 = data.rt;
          const common = Math.random() < P_COMMON;
          const commonState = 1 + trialState.a1;
          trialState.s2 = common ? commonState : 3 - commonState;
          trialState.transition = common ? "common" : "rare";
        }
        Object.assign(data, flattenTrial(trialState, schedule[trial], scheduleSeed, {
          participantCode, block, condition, trial, isPractice,
        }));
      },
    });

    trials.push({
      timeline: [
        {
          type: htmlKeyboardResponse,
          stimulus: () => `<div class="stage"><p class="too-slow">Too slow</p></div>`,
          choices: "NO_KEYS",
          trial_duration: 600,
          data: { trial_kind: "too_slow" },
        },
      ],
      conditional_function: () => trialState.missed_a1,
    });

    trials.push({
      timeline: [
        {
          type: htmlKeyboardResponse,
          stimulus: () =>
            `<div class="stage">
              <p>You arrived at the ${PLANETS[trialState.s2]?.name ?? "planet"}</p>
              ${planetHtml(trialState.s2 ?? 1)}
              <div class="row">${alienHtml(trialState.s2 ?? 1, 0)}${alienHtml(trialState.s2 ?? 1, 1)}</div>
            </div>`,
          choices: ["f", "j"],
          trial_duration: deadlines.stage2,
          response_ends_trial: true,
          data: {
            trial_kind: isPractice ? "practice_stage2" : "stage2",
            participant_code: participantCode,
            block,
            condition,
            trial,
            schedule_seed: scheduleSeed,
          },
          on_finish(data) {
            if (data.response == null) {
              trialState.missed_a2 = true;
              trialState.rt2 = deadlines.stage2;
            } else {
              trialState.a2 = data.response === "f" ? 0 : 1;
              trialState.rt2 = data.rt;
              const p = schedule[trial][trialState.s2 - 1][trialState.a2];
              trialState.r = Math.random() < p ? 1 : 0;
            }
            Object.assign(data, flattenTrial(trialState, schedule[trial], scheduleSeed, {
              participantCode, block, condition, trial, isPractice,
            }));
          },
        },
        {
          timeline: [
            {
              type: htmlKeyboardResponse,
              stimulus: () => `<div class="stage"><p class="too-slow">Too slow</p></div>`,
              choices: "NO_KEYS",
              trial_duration: 600,
              data: { trial_kind: "too_slow" },
            },
          ],
          conditional_function: () => trialState.missed_a2 && !trialState.missed_a1,
        },
        {
          timeline: [
            {
              type: htmlKeyboardResponse,
              stimulus: () =>
                `<div class="stage"><p class="feedback">${
                  trialState.r === 1 ? "Treasure!" : "No treasure"
                }</p></div>`,
              choices: "NO_KEYS",
              trial_duration: 500,
              data: { trial_kind: "feedback" },
            },
          ],
          conditional_function: () =>
            !trialState.missed_a1 && !trialState.missed_a2 && trialState.r != null,
        },
      ],
      conditional_function: () => !trialState.missed_a1,
    });
  }

  return trials;
}

function flattenTrial(state, probs, scheduleSeed, meta) {
  return {
    participant_code: meta.participantCode,
    block: meta.block,
    condition: meta.condition,
    trial: meta.trial,
    a1: state.a1,
    s2: state.s2,
    a2: state.a2,
    r: state.r,
    rt1: state.rt1,
    rt2: state.rt2,
    transition: state.transition,
    missed_a1: state.missed_a1,
    missed_a2: state.missed_a2,
    p_s1_a0: probs[0][0],
    p_s1_a1: probs[0][1],
    p_s2_a0: probs[1][0],
    p_s2_a1: probs[1][1],
    schedule_seed: scheduleSeed,
    is_practice: meta.isPractice,
  };
}

/** Collect completed experimental trial rows from jsPsych data for CSV export. */
export function collectTrialRows(jsPsych) {
  const byKey = new Map();
  for (const d of jsPsych.data.get().values()) {
    if (d.is_practice) continue;
    if (d.trial_kind !== "stage1" && d.trial_kind !== "stage2") continue;
    const key = `${d.block}:${d.trial}`;
    const prev = byKey.get(key) ?? {};
    byKey.set(key, {
      participant_code: d.participant_code ?? prev.participant_code,
      block: d.block,
      condition: d.condition,
      trial: d.trial,
      a1: d.a1 ?? prev.a1 ?? "",
      s2: d.s2 ?? prev.s2 ?? "",
      a2: d.a2 ?? prev.a2 ?? "",
      r: d.r ?? prev.r ?? "",
      rt1: d.rt1 ?? prev.rt1 ?? "",
      rt2: d.rt2 ?? prev.rt2 ?? "",
      transition: d.transition ?? prev.transition ?? "",
      missed_a1: Boolean(d.missed_a1 || prev.missed_a1),
      missed_a2: Boolean(d.missed_a2 || prev.missed_a2),
      p_s1_a0: d.p_s1_a0 ?? prev.p_s1_a0,
      p_s1_a1: d.p_s1_a1 ?? prev.p_s1_a1,
      p_s2_a0: d.p_s2_a0 ?? prev.p_s2_a0,
      p_s2_a1: d.p_s2_a1 ?? prev.p_s2_a1,
      schedule_seed: d.schedule_seed ?? prev.schedule_seed,
    });
  }
  return [...byKey.values()].sort(
    (a, b) => a.block - b.block || a.trial - b.trial,
  );
}

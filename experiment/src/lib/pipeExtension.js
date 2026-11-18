import PipeExtension from "@jspsych/extension-pipe";
import { isValidParticipantCode } from "./participantCode.js";

/** Unique, code-based filename, e.g. "TM-7F3K_session.csv". */
export function pipeFilename({ participantCode } = {}) {
  if (!isValidParticipantCode(participantCode)) throw new Error("Invalid participant code");
  return `${participantCode}_session.csv`;
}

/**
 * DataPipe extension config for `initJsPsych({ extensions: [...] })`.
 * MindBench / @jspsych/extension-pipe 0.2.0 conventions.
 */
export function buildPipeExtension({ experimentId, getFilename, simulate = false, onSave } = {}) {
  if (typeof experimentId !== "string" || experimentId.trim() === "") {
    throw new Error("DataPipe experiment ID is required");
  }
  if (typeof getFilename !== "function") throw new Error("getFilename function is required");
  return {
    type: PipeExtension,
    params: {
      experiment_id: experimentId.trim(),
      filename: () => getFilename(),
      format: "csv",
      stream: true,
      enabled: !simulate,
      on_save: (result) => onSave?.(result),
    },
  };
}

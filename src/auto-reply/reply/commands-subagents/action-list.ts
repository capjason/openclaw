import { buildSubagentList, type SubagentListItem } from "../../../agents/subagent-control.js";
import { truncateLine } from "../../../shared/subagents-format.js";
import { findLatestTaskForSessionKey } from "../../../tasks/task-registry.js";
import type { CommandHandlerResult } from "../commands-types.js";
import { type SubagentsCommandContext, RECENT_WINDOW_MINUTES, stopWithText } from "./shared.js";

function formatEntryWithDetails(entry: SubagentListItem): string {
  const parts = [entry.line];
  const task = findLatestTaskForSessionKey(entry.sessionKey);
  if (task?.progressSummary?.trim()) {
    parts.push(`   Progress: ${truncateLine(task.progressSummary.trim(), 120)}`);
  }
  if (task?.error?.trim()) {
    parts.push(`   Error: ${truncateLine(task.error.trim(), 120)}`);
  }
  return parts.join("\n");
}

export function handleSubagentsListAction(ctx: SubagentsCommandContext): CommandHandlerResult {
  const { params, runs } = ctx;
  const list = buildSubagentList({
    cfg: params.cfg,
    runs,
    recentMinutes: RECENT_WINDOW_MINUTES,
    taskMaxChars: 110,
  });
  const lines = ["active subagents:", "-----"];
  if (list.active.length === 0) {
    lines.push("(none)");
  } else {
    lines.push(list.active.map(formatEntryWithDetails).join("\n"));
  }
  lines.push("", `recent subagents (last ${RECENT_WINDOW_MINUTES}m):`, "-----");
  if (list.recent.length === 0) {
    lines.push("(none)");
  } else {
    lines.push(list.recent.map(formatEntryWithDetails).join("\n"));
  }

  return stopWithText(lines.join("\n"));
}

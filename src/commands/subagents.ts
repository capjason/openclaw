import { loadSubagentRegistryFromDisk } from "../agents/subagent-registry.store.js";
import type { SubagentRunRecord } from "../agents/subagent-registry.types.js";
import { info } from "../globals.js";
import type { RuntimeEnv } from "../runtime.js";
import { formatDurationCompact, truncateLine } from "../shared/subagents-format.js";
import { isRich, theme } from "../terminal/theme.js";

const RECENT_WINDOW_MINUTES = 30;

function resolveStatus(entry: SubagentRunRecord): string {
  if (!entry.endedAt) {
    return "running";
  }
  const status = entry.outcome?.status ?? "done";
  if (status === "ok") {
    return "done";
  }
  if (status === "error") {
    return "failed";
  }
  return status;
}

function formatRunDetail(entry: SubagentRunRecord, index: number, rich: boolean): string[] {
  const now = Date.now();
  const runtimeMs = entry.endedAt
    ? entry.endedAt - (entry.startedAt ?? entry.createdAt) + (entry.accumulatedRuntimeMs ?? 0)
    : now - (entry.startedAt ?? entry.createdAt) + (entry.accumulatedRuntimeMs ?? 0);

  const label = truncateLine(entry.label || entry.task.split("\n")[0].trim(), 48);
  const status = resolveStatus(entry);
  const runtime = formatDurationCompact(runtimeMs);
  const model = entry.model ?? "n/a";

  const statusFormatted = !rich
    ? status
    : status === "running"
      ? theme.accentBright(status)
      : status === "failed"
        ? theme.error(status)
        : status === "done"
          ? theme.success(status)
          : theme.muted(status);

  const lines: string[] = [];
  lines.push(`  ${index}. ${label}  [${statusFormatted}]  ${runtime}  ${model}`);
  lines.push(`     Task: ${truncateLine(entry.task.replace(/\s+/g, " ").trim(), 120)}`);

  if (entry.frozenResultText?.trim()) {
    const resultLabel = rich ? theme.accentBright("Result") : "Result";
    lines.push(
      `     ${resultLabel}: ${truncateLine(entry.frozenResultText.trim().replace(/\s+/g, " "), 120)}`,
    );
  }

  if (entry.outcome?.status === "error") {
    const error =
      "error" in entry.outcome ? (entry.outcome as { error?: string }).error : undefined;
    if (error) {
      const errorLabel = rich ? theme.error("Error") : "Error";
      lines.push(`     ${errorLabel}: ${truncateLine(error, 120)}`);
    }
  }

  lines.push("");
  return lines;
}

export async function subagentsListCommand(
  opts: { json?: boolean; all?: boolean; session?: string; limit?: number },
  runtime: RuntimeEnv,
) {
  const runs = loadSubagentRegistryFromDisk();
  const now = Date.now();
  const recentCutoff = now - RECENT_WINDOW_MINUTES * 60_000;

  const allRuns = [...runs.values()].toSorted((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));

  // Filter by session if specified
  const filtered = opts.session
    ? allRuns.filter(
        (r) =>
          r.requesterSessionKey === opts.session ||
          r.childSessionKey === opts.session ||
          r.requesterSessionKey.includes(opts.session!),
      )
    : allRuns;

  const active = filtered.filter((r) => !r.endedAt);
  const recent = filtered.filter((r) => r.endedAt && r.endedAt >= recentCutoff);
  const older = filtered.filter((r) => r.endedAt && r.endedAt < recentCutoff);

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          activeCount: active.length,
          recentCount: recent.length,
          totalCount: filtered.length,
          active,
          recent,
          older,
        },
        null,
        2,
      ),
    );
    return;
  }

  const rich = isRich();
  const heading = rich ? theme.heading : (s: string) => s;

  runtime.log(info(`Subagent runs (from disk registry, ${filtered.length} total)`));
  runtime.log("");

  runtime.log(heading(`Active (${active.length}):`));
  if (active.length === 0) {
    runtime.log("  (none)");
    runtime.log("");
  } else {
    let idx = 1;
    for (const entry of active) {
      for (const line of formatRunDetail(entry, idx, rich)) {
        runtime.log(line);
      }
      idx += 1;
    }
  }

  runtime.log(heading(`Recent (last ${RECENT_WINDOW_MINUTES}m, ${recent.length}):`));
  if (recent.length === 0) {
    runtime.log("  (none)");
    runtime.log("");
  } else {
    let idx = 1;
    for (const entry of recent) {
      for (const line of formatRunDetail(entry, idx, rich)) {
        runtime.log(line);
      }
      idx += 1;
    }
  }

  const limit = opts.limit ?? 5;
  const shown = older.slice(0, limit);
  runtime.log(heading(`Older (${older.length} total, showing ${shown.length}):`));
  if (shown.length === 0) {
    runtime.log("  (none)");
    runtime.log("");
  } else {
    let idx = 1;
    for (const entry of shown) {
      for (const line of formatRunDetail(entry, idx, rich)) {
        runtime.log(line);
      }
      idx += 1;
    }
  }
}

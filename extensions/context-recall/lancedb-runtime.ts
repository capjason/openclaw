/**
 * LanceDB runtime loader for context-recall plugin.
 *
 * Reuses the same pattern as memory-lancedb: try bundled import first,
 * fall back to installing the native module in a plugin-runtime directory.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

type LanceDbModule = typeof import("@lancedb/lancedb");

type LanceDbRuntimeLogger = {
  info?: (message: string) => void;
  warn?: (message: string) => void;
};

type RuntimeManifest = {
  name: string;
  private: true;
  type: "module";
  dependencies: Record<string, string>;
};

function resolveStateDir(): string {
  const envDir = process.env.OPENCLAW_STATE_DIR?.trim();
  if (envDir) return envDir;
  const home = process.env.HOME?.trim() || os.homedir();
  return path.join(home, ".openclaw");
}

function resolveLanceDbSpec(): string {
  const manifestPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "package.json");
  try {
    const pkg = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as {
      dependencies?: Record<string, string>;
    };
    const spec = pkg?.dependencies?.["@lancedb/lancedb"];
    if (spec) return spec;
  } catch {
    // fall through
  }
  return "^0.27.1";
}

const RUNTIME_MANIFEST: RuntimeManifest = {
  name: "openclaw-context-recall-lancedb-runtime",
  private: true,
  type: "module",
  dependencies: {
    "@lancedb/lancedb": resolveLanceDbSpec(),
  },
};

function resolveRuntimeDir(): string {
  return path.join(resolveStateDir(), "plugin-runtimes", "context-recall", "lancedb");
}

function readManifest(filePath: string): RuntimeManifest | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as RuntimeManifest;
  } catch {
    return null;
  }
}

function manifestsMatch(actual: RuntimeManifest | null, expected: RuntimeManifest): boolean {
  if (!actual) return false;
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function resolveEntry(runtimeDir: string): string | null {
  const pkgPath = path.join(runtimeDir, "package.json");
  if (!manifestsMatch(readManifest(pkgPath), RUNTIME_MANIFEST)) return null;
  try {
    const req = createRequire(pkgPath);
    return req.resolve("@lancedb/lancedb");
  } catch {
    return null;
  }
}

function collectSpawnOutput(params: {
  command: string;
  args: string[];
  cwd: string;
}): Promise<{ code: number | null; stdout: string; stderr: string; error?: Error }> {
  return new Promise((resolve) => {
    const child = spawn(params.command, params.args, {
      cwd: params.cwd,
      env: process.env,
      shell: process.platform === "win32",
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer | string) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer | string) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => resolve({ code: null, stdout, stderr, error }));
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

async function installRuntime(runtimeDir: string, logger?: LanceDbRuntimeLogger): Promise<string> {
  const pkgPath = path.join(runtimeDir, "package.json");
  if (!manifestsMatch(readManifest(pkgPath), RUNTIME_MANIFEST)) {
    await fs.promises.rm(path.join(runtimeDir, "node_modules"), { recursive: true, force: true });
    await fs.promises.rm(path.join(runtimeDir, "package-lock.json"), { force: true });
  }

  await fs.promises.mkdir(runtimeDir, { recursive: true });
  await fs.promises.writeFile(pkgPath, `${JSON.stringify(RUNTIME_MANIFEST, null, 2)}\n`, "utf8");

  const result = await collectSpawnOutput({
    command: "npm",
    args: ["install", "--omit=dev", "--silent", "--ignore-scripts", "--package-lock=false"],
    cwd: runtimeDir,
  });

  if (result.error) {
    const spawnError = result.error as NodeJS.ErrnoException;
    throw new Error(
      spawnError.code === "ENOENT"
        ? "npm is required to install the LanceDB runtime but was not found on PATH"
        : result.error.message,
    );
  }
  if ((result.code ?? 0) !== 0) {
    const detail = result.stderr.trim() || result.stdout.trim();
    throw new Error(detail || `npm exited with code ${result.code ?? "unknown"}`);
  }

  const resolved = resolveEntry(runtimeDir);
  if (!resolved) {
    throw new Error("installed LanceDB runtime is missing the @lancedb/lancedb entry");
  }
  logger?.info?.(`context-recall: installed LanceDB runtime under ${runtimeDir}`);
  return resolved;
}

let loadPromise: Promise<LanceDbModule> | null = null;

export async function loadLanceDbModule(logger?: LanceDbRuntimeLogger): Promise<LanceDbModule> {
  if (!loadPromise) {
    loadPromise = (async () => {
      // Try bundled import first
      try {
        return await import("@lancedb/lancedb");
      } catch (bundledError) {
        const runtimeDir = resolveRuntimeDir();

        // Check existing runtime install
        const existing = resolveEntry(runtimeDir);
        if (existing) {
          try {
            return await import(pathToFileURL(existing).href);
          } catch {
            // stale, reinstall below
          }
        }

        if (process.env.OPENCLAW_NIX_MODE === "1") {
          throw new Error(
            `context-recall: failed to load LanceDB and Nix mode disables auto-install. ${String(bundledError)}`,
          );
        }

        logger?.warn?.(
          `context-recall: bundled LanceDB unavailable; installing under ${runtimeDir}`,
        );
        const installedEntry = await installRuntime(runtimeDir, logger);
        try {
          return await import(pathToFileURL(installedEntry).href);
        } catch (runtimeError) {
          throw new Error(
            `context-recall: failed to load LanceDB after install. ${String(runtimeError)}`,
          );
        }
      }
    })().catch((error) => {
      loadPromise = null;
      throw error;
    });
  }
  return await loadPromise;
}

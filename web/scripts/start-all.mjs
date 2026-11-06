#!/usr/bin/env node

import { spawn, execSync } from "child_process";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(webRoot, "..");

const isDev = process.argv.includes("--dev");

// 1. Locate Python Interpreter
let pythonPath = path.join(repoRoot, ".venv", "Scripts", "python.exe");
if (!fs.existsSync(pythonPath)) {
  pythonPath = path.join(repoRoot, ".venv", "bin", "python");
  if (!fs.existsSync(pythonPath)) {
    pythonPath = "python";
  }
}

const backendScript = path.join(webRoot, "mini-services", "inference_service.py");
const nextBin = path.join(webRoot, "node_modules", "next", "dist", "bin", "next");

console.log("\n=======================================================");
console.log("  GeoCaption Application: Launching Full Stack");
console.log(`  Mode:     ${isDev ? "Development (Hot Reload)" : "Production"}`);
console.log(`  Frontend: http://localhost:3000`);
console.log(`  Backend:  http://127.0.0.1:5001 (PyTorch NWPUCaptioner)`);
console.log(`  Python:   ${pythonPath}`);
console.log("=======================================================\n");

let backendProc = null;
let frontendProc = null;
let isShuttingDown = false;

function cleanShutdown() {
  if (isShuttingDown) return;
  isShuttingDown = true;
  console.log("\n[GeoCaption] Stopping full-stack servers...");

  const killChild = (child) => {
    if (!child || !child.pid) return;
    try {
      if (process.platform === "win32") {
        execSync(`taskkill /pid ${child.pid} /T /F`, { stdio: "ignore" });
      } else {
        child.kill("SIGTERM");
      }
    } catch {
      /* process already terminated */
    }
  };

  killChild(backendProc);
  killChild(frontendProc);
  process.exit(0);
}

process.on("SIGINT", cleanShutdown);
process.on("SIGTERM", cleanShutdown);

// 2. Build if in production start mode and .next missing
const nextFolder = path.join(webRoot, ".next");
if (!isDev && !fs.existsSync(nextFolder)) {
  console.log("[GeoCaption] Building Next.js production bundle...");
  execSync(`"${process.execPath}" "${nextBin}" build --webpack`, {
    cwd: webRoot,
    stdio: "inherit",
  });
}

// 3. Start Backend (PyTorch NWPUCaptioner)
console.log("[GeoCaption] Starting PyTorch Inference Service on port 5001...");
backendProc = spawn(pythonPath, ["-u", backendScript], {
  cwd: repoRoot,
  stdio: "inherit",
  shell: false,
  env: {
    ...process.env,
    PYTHONPATH: `${path.join(repoRoot, "src")};${path.join(repoRoot, "scripts")}`,
    PYTHONUNBUFFERED: "1",
  },
});

// 4. Start Frontend (Next.js)
const frontendArgs = isDev
  ? [nextBin, "dev", "--webpack", "-p", "3000"]
  : [nextBin, "start", "-p", "3000"];

console.log(`[GeoCaption] Starting Next.js Web App on port 3000...`);
frontendProc = spawn(process.execPath, frontendArgs, {
  cwd: webRoot,
  stdio: "inherit",
  shell: false,
  env: { ...process.env },
});

frontendProc.on("exit", (code) => {
  if (!isShuttingDown) {
    console.log(`[GeoCaption] Frontend exited (code: ${code})`);
    cleanShutdown();
  }
});

backendProc.on("exit", (code) => {
  if (!isShuttingDown) {
    console.log(`[GeoCaption] Backend exited (code: ${code})`);
    cleanShutdown();
  }
});

// Keep parent event loop active
setInterval(() => {}, 60000);

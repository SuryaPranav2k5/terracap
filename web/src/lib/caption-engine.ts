import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import type { AttentionStep, Keyword } from "@/lib/types";

export interface ParsedCaption {
  caption: string;
  tokens: string[];
  keywords: Keyword[];
  confidence: number;
  attentionSteps: AttentionStep[];
  genTimeMs: number;
}

const MICROSERVICE_URL = "http://127.0.0.1:5001/predict";

/**
 * Runs genuine PyTorch model inference (NWPUCaptioner: ConvNeXt-Tiny + LoRA + Spatial Attention).
 * Connects to the local Python microservice if running, or spawns the Python inference CLI as a fallback.
 */
export async function runCaptionModel(
  imageDataUrl: string
): Promise<ParsedCaption> {
  const started = Date.now();

  // Strategy 1: Attempt to use the resident Python microservice (sub-second latency)
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(MICROSERVICE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl, beam_size: 3 }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.ok) {
      const data = await res.json();
      return {
        caption: data.caption,
        tokens: data.tokens ?? [],
        keywords: data.keywords ?? [],
        confidence: data.confidence ?? 0.92,
        attentionSteps: data.attention_steps ?? [],
        genTimeMs: data.genTimeMs ?? Date.now() - started,
      };
    }
  } catch {
    // Microservice is not running, proceed to CLI execution fallback
  }

  // Strategy 2: Direct Python CLI execution via .venv/Scripts/python.exe
  const repoRoot = path.resolve(process.cwd(), "..");
  let pythonPath = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  if (!fs.existsSync(pythonPath)) {
    pythonPath = path.join(repoRoot, ".venv", "bin", "python");
    if (!fs.existsSync(pythonPath)) {
      pythonPath = "python";
    }
  }

  const scriptPath = path.join(repoRoot, "scripts", "infer.py");

  // Write base64 to a temporary file to avoid Windows command line character limit
  const base64Data = imageDataUrl.replace(/^data:image\/\w+;base64,/, "");
  const tempFilePath = path.join(
    os.tmpdir(),
    `geocaption-${Date.now()}-${Math.random().toString(36).slice(2)}.png`
  );

  fs.writeFileSync(tempFilePath, Buffer.from(base64Data, "base64"));

  return new Promise((resolve, reject) => {
    const pyProcess = spawn(
      pythonPath,
      [scriptPath, "--image", tempFilePath, "--beam_size", "3"],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          PYTHONPATH: path.join(repoRoot, "src"),
        },
      }
    );

    let stdoutData = "";
    let stderrData = "";

    pyProcess.stdout.on("data", (chunk) => {
      stdoutData += chunk.toString();
    });

    pyProcess.stderr.on("data", (chunk) => {
      stderrData += chunk.toString();
    });

    pyProcess.on("close", (code) => {
      try {
        if (fs.existsSync(tempFilePath)) {
          fs.unlinkSync(tempFilePath);
        }
      } catch {
        /* ignore temp cleanup error */
      }

      if (code !== 0) {
        console.error("[runCaptionModel] Python process failed:", stderrData);
        return reject(
          new Error(
            `Inference engine failed: ${stderrData || `exit code ${code}`}`
          )
        );
      }

      try {
        // Find JSON payload in stdout
        const trimmed = stdoutData.trim();
        const jsonStart = trimmed.indexOf("{");
        const jsonEnd = trimmed.lastIndexOf("}");
        if (jsonStart === -1 || jsonEnd === -1) {
          throw new Error("No JSON found in model output");
        }
        const jsonStr = trimmed.slice(jsonStart, jsonEnd + 1);
        const data = JSON.parse(jsonStr);

        resolve({
          caption: data.caption,
          tokens: data.tokens ?? [],
          keywords: data.keywords ?? [],
          confidence: data.confidence ?? 0.92,
          attentionSteps: data.attention_steps ?? [],
          genTimeMs: Date.now() - started,
        });
      } catch (parseErr) {
        console.error(
          "[runCaptionModel] Failed to parse model output:",
          stdoutData
        );
        reject(
          new Error(
            `Failed to parse model response: ${
              parseErr instanceof Error ? parseErr.message : String(parseErr)
            }`
          )
        );
      }
    });

    pyProcess.on("error", (err) => {
      try {
        if (fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);
      } catch {
        /* ignore */
      }
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
}

import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { runCaptionModel } from "@/lib/caption-engine";
import { addSessionUpload } from "@/lib/session-store";
import type { GenerationItem } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 120;

const MAX_BODY_CHARS = 20 * 1024 * 1024; // ~15MB binary after base64

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    let image: unknown = body?.image;
    const imageName: string =
      typeof body?.imageName === "string" ? body.imageName : "upload.png";

    // Handle relative public paths (e.g. /images/stadium.png)
    if (typeof image === "string" && image.startsWith("/")) {
      const publicPath = path.join(process.cwd(), "public", image.replace(/^\//, ""));
      if (fs.existsSync(publicPath)) {
        const ext = path.extname(publicPath).slice(1) || "png";
        const buffer = fs.readFileSync(publicPath);
        image = `data:image/${ext};base64,${buffer.toString("base64")}`;
      }
    }

    if (
      typeof image !== "string" ||
      (!image.startsWith("data:image/") && !image.startsWith("http"))
    ) {
      return NextResponse.json(
        { error: "A valid image data URL is required." },
        { status: 400 }
      );
    }
    if (image.length > MAX_BODY_CHARS) {
      return NextResponse.json(
        { error: "Image is too large. Please use an image under 15 MB." },
        { status: 413 }
      );
    }

    const {
      caption,
      tokens,
      keywords,
      confidence,
      attentionSteps,
      genTimeMs,
    } = await runCaptionModel(image);

    const record: GenerationItem = {
      id: `gen-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      imageSrc: image,
      imageName,
      label: "User Upload",
      caption,
      tokens,
      keywords,
      confidence,
      attentionSteps,
      genTimeMs,
      source: "upload",
      createdAt: new Date().toISOString(),
    };

    addSessionUpload(record);

    return NextResponse.json(record);
  } catch (err) {
    console.error("[/api/caption] inference error:", err);
    return NextResponse.json(
      {
        error:
          err instanceof Error
            ? err.message
            : "The caption model failed to process this image. Please try again.",
      },
      { status: 500 }
    );
  }
}

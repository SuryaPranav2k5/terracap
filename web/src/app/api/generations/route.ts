import { NextRequest, NextResponse } from "next/server";
import { REFERENCE_GENERATIONS } from "@/lib/reference-data";
import { getSessionUploads } from "@/lib/session-store";
import type { GenerationItem } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const limit = Math.min(
      100,
      Math.max(1, Number(searchParams.get("limit")) || 24)
    );
    const offset = Math.max(0, Number(searchParams.get("offset")) || 0);
    const source = searchParams.get("source"); // "reference" | "upload" | undefined

    const uploads = getSessionUploads();
    let all: GenerationItem[] = [...uploads, ...REFERENCE_GENERATIONS];

    if (source === "reference") {
      all = REFERENCE_GENERATIONS;
    } else if (source === "upload") {
      all = uploads;
    }

    const items = all.slice(offset, offset + limit);

    return NextResponse.json({
      items,
      total: all.length,
    });
  } catch (err) {
    console.error("[/api/generations GET] failed:", err);
    return NextResponse.json(
      { error: "Failed to load generations." },
      { status: 500 }
    );
  }
}

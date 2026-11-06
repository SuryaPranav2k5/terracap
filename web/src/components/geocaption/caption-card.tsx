"use client";

import {
  FileText,
  Copy,
  Download,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ConfidenceRing } from "@/components/geocaption/confidence-ring";
import { KeywordBars } from "@/components/geocaption/keyword-bars";
import { useGenStore } from "@/lib/store";
import { useTypewriter } from "@/hooks/use-typewriter";
import { formatSeconds } from "@/lib/format";
import { copyTextToClipboard } from "@/lib/clipboard";

export function CaptionCard() {
  const result = useGenStore((s) => s.result);
  const status = useGenStore((s) => s.status);
  const error = useGenStore((s) => s.error);
  const generate = useGenStore((s) => s.generate);

  const { displayed } = useTypewriter(result?.caption ?? "", 3);
  const generating = status === "generating";

  const copyCaption = async () => {
    if (!result) return;
    const ok = await copyTextToClipboard(result.caption);
    if (ok) {
      toast.success("Caption copied to clipboard");
    } else {
      toast.error("Could not copy — clipboard unavailable.");
    }
  };

  const downloadCaption = () => {
    if (!result) return;
    const content = [
      "GeoCaption — Generated Caption",
      "================================",
      "",
      result.caption,
      "",
      "Keywords:",
      ...result.keywords.map((k) => `  ${k.word}: ${k.score.toFixed(2)}`),
      "",
      `Confidence: ${Math.round(result.confidence * 100)}%`,
      `Generation time: ${formatSeconds(result.genTimeMs)}`,
      `Source image: ${result.imageName}`,
      `Created: ${new Date(result.createdAt).toLocaleString()}`,
    ].join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `geocaption-${result.id.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Caption downloaded");
  };

  return (
    <Card className="flex h-full flex-col justify-between rounded-2xl border border-white/10 bg-[#07090e]/90 p-4 sm:p-5 backdrop-blur-md shadow-md min-h-[310px]">
      <CardHeader className="p-0 pb-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-500/15 text-amber-400">
            <FileText className="h-4 w-4" strokeWidth={2} />
          </span>
          <CardTitle className="text-[15px] font-semibold text-white">
            Generated Caption
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-between p-0 gap-3">
        {/* Caption text area */}
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5 min-h-[80px] flex items-center shadow-inner">
          {generating ? (
            <div className="flex w-full flex-col gap-2" aria-live="polite">
              <div className="shimmer h-3 w-full rounded" />
              <div className="shimmer h-3 w-[92%] rounded" />
              <p className="flex items-center gap-1.5 text-[12px] text-amber-400 mt-1">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Analyzing image with the captioning model…
              </p>
            </div>
          ) : status === "error" ? (
            <div className="flex w-full flex-col items-start gap-1.5">
              <p className="flex items-center gap-1.5 text-[12px] text-rose-400">
                <AlertTriangle className="h-3.5 w-3.5" />
                {error ?? "Failed to generate caption."}
              </p>
              <button
                onClick={() => void generate()}
                className="text-[11px] text-amber-400 hover:underline"
              >
                Retry generation
              </button>
            </div>
          ) : (
            <p className={`text-[13.5px] leading-relaxed ${result ? "text-white/95 font-medium" : "text-white/40 italic"}`}>
              {displayed || "No caption generated yet. Select or upload an image and click 'Generate Caption'."}
            </p>
          )}
        </div>

          {/* Action row: Copy, Download */}
          <div className="flex items-center gap-2">
            <button
              onClick={copyCaption}
              disabled={!result || generating}
              className="inline-flex h-7.5 sm:h-8 items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-3 text-[11px] font-medium text-white/80 transition-colors hover:border-amber-400/40 hover:text-white disabled:opacity-40"
            >
              <Copy className="h-3.5 w-3.5" />
              Copy
            </button>
            <button
              onClick={downloadCaption}
              disabled={!result || generating}
              className="inline-flex h-7.5 sm:h-8 items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-3 text-[11px] font-medium text-white/80 transition-colors hover:border-amber-400/40 hover:text-white disabled:opacity-40"
            >
              <Download className="h-3.5 w-3.5" />
              Download
            </button>
          </div>

          {/* Bottom row: Circular Gauge + Keyword Confidence Bars */}
          <div className="flex items-center gap-4 pt-2.5 border-t border-white/[0.08]">
            <div className="shrink-0 flex items-center justify-center">
              <ConfidenceRing
                confidence={result?.confidence ?? 0}
                size={66}
                stroke={5.5}
              />
            </div>
            <div className="flex-1 min-w-0">
              <KeywordBars keywords={result?.keywords} />
            </div>
          </div>
      </CardContent>
    </Card>
  );
}

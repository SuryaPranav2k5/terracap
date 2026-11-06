"use client";

import { useState } from "react";
import Image from "next/image";
import { Copy, Download, Trash2 } from "lucide-react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ConfidenceRing } from "@/components/geocaption/confidence-ring";
import { KeywordBars } from "@/components/geocaption/keyword-bars";
import type { GenerationItem } from "@/lib/types";
import { fullDate, formatSeconds, timeAgo } from "@/lib/format";
import { copyTextToClipboard } from "@/lib/clipboard";

export function GenerationDetailDialog({
  generation,
  open,
  onOpenChange,
  onDeleted,
}: {
  generation: GenerationItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeleted?: () => void;
}) {
  const [deleting, setDeleting] = useState(false);
  if (!generation) return null;

  const copyCaption = async () => {
    const ok = await copyTextToClipboard(generation.caption);
    if (ok) {
      toast.success("Caption copied to clipboard");
    } else {
      toast.error("Could not copy — clipboard unavailable.");
    }
  };

  const downloadCaption = () => {
    const content = [
      "GeoCaption — Generated Caption",
      "================================",
      "",
      generation.caption,
      "",
      "Keywords:",
      ...generation.keywords.map((k) => `  ${k.word}: ${k.score.toFixed(2)}`),
      "",
      `Confidence: ${Math.round(generation.confidence * 100)}%`,
      `Generation time: ${formatSeconds(generation.genTimeMs)}`,
      `Source image: ${generation.imageName}`,
      `Created: ${fullDate(generation.createdAt)}`,
    ].join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `geocaption-${generation.id.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const deleteGeneration = async () => {
    setDeleting(true);
    try {
      const res = await fetch(`/api/generations?id=${generation.id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error();
      toast.success("Generation deleted");
      onOpenChange(false);
      onDeleted?.();
    } catch {
      toast.error("Failed to delete generation.");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="gc-scroll max-h-[88vh] max-w-2xl overflow-y-auto border-gc-border bg-gc-panel-solid">
        <DialogHeader>
          <DialogTitle className="font-display text-gc-gold">
            {generation.label ?? generation.imageName}
          </DialogTitle>
          <DialogDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12.5px] text-gc-faint">
            <span className="font-mono">{fullDate(generation.createdAt)}</span>
            <span>·</span>
            <span>{timeAgo(generation.createdAt)}</span>
            <span>·</span>
            <span className="rounded-full border border-gc-border px-2 py-0.5 text-[10.5px] uppercase tracking-wider">
              {generation.source === "upload" ? "Live upload" : "Reference set"}
            </span>
            <span>·</span>
            <span className="font-mono">
              {formatSeconds(generation.genTimeMs)}
            </span>
          </DialogDescription>
        </DialogHeader>

        <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-gc-border">
          <Image
            src={generation.imageSrc}
            alt={`Satellite image: ${generation.caption.slice(0, 80)}`}
            fill
            unoptimized={generation.imageSrc.startsWith("data:")}
            sizes="90vw"
            className="object-cover"
          />
        </div>

        <p className="text-[15px] leading-[1.7] text-gc-text/90">
          {generation.caption}
        </p>

        <div className="flex items-center gap-5 rounded-lg border border-gc-border bg-gc-panel p-4">
          <div className="flex flex-col items-center gap-1.5">
            <ConfidenceRing confidence={generation.confidence} size={76} stroke={7} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-gc-faint">
              Confidence
            </span>
          </div>
          <div className="min-w-0 flex-1">
            <KeywordBars keywords={generation.keywords} />
          </div>
        </div>

        <div className="flex items-center justify-between gap-2">
          <div className="flex gap-2.5">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void copyCaption()}
              className="gap-1.5 border-gc-border text-gc-dim hover:border-gc-gold/50 hover:text-gc-gold"
            >
              <Copy className="h-3.5 w-3.5" /> Copy
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={downloadCaption}
              className="gap-1.5 border-gc-border text-gc-dim hover:border-gc-gold/50 hover:text-gc-gold"
            >
              <Download className="h-3.5 w-3.5" /> Download
            </Button>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void deleteGeneration()}
            disabled={deleting}
            className="gap-1.5 border-red-500/30 text-red-500 hover:border-red-500/60 hover:bg-red-500/10 hover:text-red-500"
          >
            <Trash2 className="h-3.5 w-3.5" />
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

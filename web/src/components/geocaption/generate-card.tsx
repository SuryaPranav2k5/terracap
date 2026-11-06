"use client";

import { useState } from "react";
import Image from "next/image";
import { Box, Sparkles, ArrowRight, Loader2, Maximize2, Image as ImageIcon } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { UploadZone } from "@/components/geocaption/upload-zone";
import { useGenStore } from "@/lib/store";

export function GenerateCard() {
  const image = useGenStore((s) => s.image);
  const result = useGenStore((s) => s.result);
  const status = useGenStore((s) => s.status);
  const generate = useGenStore((s) => s.generate);
  const generating = status === "generating";
  const [expanded, setExpanded] = useState(false);

  const displaySrc = image?.dataUrl ?? result?.imageSrc ?? null;

  return (
    <Card className="flex flex-col justify-between rounded-2xl border border-white/10 bg-[#07090e]/90 p-4 sm:p-5 backdrop-blur-md shadow-md min-h-[310px]">
      <CardHeader className="p-0 pb-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-500/15 text-amber-400">
            <Box className="h-4 w-4" strokeWidth={2} />
          </span>
          <CardTitle className="text-[15px] font-semibold text-white">
            Generate Caption
          </CardTitle>
        </div>
        <p className="text-[12px] text-white/60 leading-tight mt-0.5">
          Upload a satellite or aerial image to generate a detailed description.
        </p>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-between p-0 gap-3.5">
        {/* Two columns: Drop Zone & Input Image preview */}
        <div className="grid grid-cols-2 gap-3 items-stretch min-h-[200px] sm:min-h-[220px]">
          <div className="h-full">
            <UploadZone />
          </div>

          {/* Input image preview / Empty state */}
          {displaySrc ? (
            <div className="relative h-full min-h-[200px] sm:min-h-[220px] overflow-hidden rounded-xl border border-white/10 bg-black/60 shadow-inner">
              <Image
                src={displaySrc}
                alt="Input satellite image"
                fill
                unoptimized={displaySrc.startsWith("data:")}
                sizes="320px"
                className="object-cover"
              />
              <span className="absolute left-2 top-2 rounded-md bg-black/80 px-2 py-0.5 text-[10px] font-medium text-white/95 backdrop-blur-md border border-white/10">
                Input Image
              </span>
              <button
                onClick={() => setExpanded(true)}
                aria-label="Expand image"
                className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-md bg-black/75 text-white/80 backdrop-blur-sm transition-colors hover:text-white hover:bg-black/90 border border-white/10"
              >
                <Maximize2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <div className="relative flex h-full min-h-[200px] sm:min-h-[220px] flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-white/[0.01] p-3 text-center select-none">
              <ImageIcon className="h-8 w-8 text-white/20 mb-2" strokeWidth={1.5} />
              <p className="text-[12px] font-medium text-white/60 leading-tight">
                No image selected
              </p>
              <p className="text-[10px] text-white/35 leading-tight mt-1">
                Drop an image or select a sample
              </p>
            </div>
          )}
        </div>

        {/* Generate Button cleanly anchored below */}
        <button
          onClick={() => void generate()}
          disabled={!displaySrc || generating}
          className={`relative flex h-11 w-full items-center justify-between rounded-xl px-4 text-[13px] font-semibold transition-all shrink-0 ${
            displaySrc && !generating
              ? "bg-gradient-to-r from-[#ffe49e] via-[#e5a84b] to-[#c98a2c] text-zinc-950 shadow-lg hover:brightness-105 active:scale-[0.99]"
              : "bg-white/5 border border-white/10 text-white/40 cursor-not-allowed"
          }`}
        >
          {generating ? (
            <span className="flex items-center gap-2 mx-auto text-zinc-950 font-semibold">
              <Loader2 className="h-4 w-4 animate-spin text-zinc-950" />
              Generating Caption with NWPUCaptioner…
            </span>
          ) : !displaySrc ? (
            <span className="flex items-center gap-1.5 mx-auto text-white/40 font-medium">
              Select or Upload Image to Generate
            </span>
          ) : (
            <>
              <span className="flex items-center gap-2 text-zinc-950 font-bold">
                <Sparkles className="h-4 w-4 fill-zinc-950/20" />
                Generate Caption
              </span>
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-black/85 text-amber-300">
                <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </>
          )}
        </button>
      </CardContent>

      {/* Expanded Image Dialog */}
      {displaySrc && (
        <Dialog open={expanded} onOpenChange={setExpanded}>
          <DialogContent className="max-w-3xl border-white/10 bg-black/90 p-4">
            <DialogHeader>
              <DialogTitle className="text-sm font-semibold text-white">
                Input Remote Sensing Image
              </DialogTitle>
              <DialogDescription className="text-xs text-white/60">
                Full resolution inspection
              </DialogDescription>
            </DialogHeader>
            <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-white/10">
              <Image
                src={displaySrc}
                alt="Expanded preview"
                fill
                unoptimized={displaySrc.startsWith("data:")}
                className="object-contain"
              />
            </div>
          </DialogContent>
        </Dialog>
      )}
    </Card>
  );
}

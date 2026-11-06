"use client";

import Image from "next/image";
import {
  Boxes,
  Cpu,
  Layers,
  ChevronDown,
  Brain,
  Code,
  Image as ImageIcon,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useGenStore } from "@/lib/store";

function Connector() {
  return (
    <div className="flex justify-center py-0 text-white/30">
      <ChevronDown className="h-3 w-3" />
    </div>
  );
}

export function ArchitectureCard() {
  const image = useGenStore((s) => s.image);
  const result = useGenStore((s) => s.result);
  const displaySrc = image?.dataUrl ?? result?.imageSrc ?? null;

  return (
    <Card className="flex h-full flex-col justify-between rounded-xl border border-white/10 bg-[#07090e]/90 p-2.5 backdrop-blur-md shadow-md min-h-[362px]">
      <CardHeader className="p-0 pb-1.5">
        <div className="flex items-center gap-1.5">
          <span className="flex h-5 w-5 items-center justify-center rounded-md bg-amber-500/15 text-amber-400">
            <Boxes className="h-3 w-3" strokeWidth={2} />
          </span>
          <CardTitle className="text-[12.5px] font-semibold text-white">
            Model Architecture
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-between p-0 gap-0">
        {/* Node 1: Input Image with thumbnail */}
        <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] p-1.5">
          <div className="relative h-6.5 w-6.5 shrink-0 overflow-hidden rounded border border-white/15 bg-black/50 flex items-center justify-center">
            {displaySrc ? (
              <Image
                src={displaySrc}
                alt="Input image thumbnail"
                fill
                unoptimized={displaySrc.startsWith("data:")}
                className="object-cover"
              />
            ) : (
              <ImageIcon className="h-3.5 w-3.5 text-white/30" />
            )}
          </div>
          <div className="leading-tight">
            <p className="text-[10px] font-semibold text-white/90">
              {displaySrc ? "Input Image" : "Input Image (Pending)"}
            </p>
            <p className="font-mono text-[8.5px] text-white/50">256 × 256 × 3</p>
          </div>
        </div>

        <Connector />

        {/* Node 2: ConvNeXt-Tiny (Frozen) */}
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/35 bg-emerald-500/[0.08] p-1.5 text-emerald-300">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-emerald-500/20 text-emerald-400">
            <Cpu className="h-3 w-3" />
          </span>
          <div className="leading-tight">
            <p className="text-[10px] font-semibold text-emerald-200">
              ConvNeXt-Tiny (Frozen)
            </p>
            <p className="font-mono text-[8.5px] text-emerald-400/80">27.8M parameters</p>
          </div>
        </div>

        <Connector />

        {/* Node 3: LoRA Adapters (Trainable) */}
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/[0.08] p-1.5 text-amber-200">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-amber-500/20 text-amber-400">
            <Layers className="h-3 w-3" />
          </span>
          <div className="leading-tight">
            <p className="text-[10px] font-semibold text-amber-200">
              LoRA Adapters (Trainable)
            </p>
            <p className="font-mono text-[8.5px] text-amber-400/80">1.06M parameters</p>
          </div>
        </div>

        <Connector />

        {/* Node 4: Visual Feature Grid */}
        <div className="rounded-lg border border-sky-500/30 bg-sky-500/[0.06] p-1.5 text-center leading-tight">
          <p className="text-[10px] font-semibold text-sky-200">
            Visual Feature Grid
          </p>
          <p className="font-mono text-[8.5px] text-sky-400/80">
            64 tokens (8 × 8) × 768
          </p>
        </div>

        <Connector />

        {/* Node 5: Spatial Attention */}
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/[0.06] p-1.5 text-center leading-tight">
          <p className="text-[10px] font-semibold text-rose-200">
            Spatial Attention
          </p>
          <p className="font-mono text-[8.5px] text-rose-400/80">
            (Additive Bahdanau)
          </p>
        </div>

        <Connector />

        {/* Node 6: Context Gating */}
        <div className="flex items-center justify-center gap-1.5 rounded-lg border border-purple-500/30 bg-purple-500/[0.06] p-1.5 text-center leading-tight">
          <Brain className="h-3 w-3 text-purple-400 shrink-0" />
          <p className="text-[10px] font-semibold text-purple-200">
            Context Gating (β-Gate)
          </p>
        </div>

        <Connector />

        {/* Node 7: LSTM Decoder */}
        <div className="rounded-lg border border-blue-500/35 bg-blue-500/[0.08] p-1.5 text-center leading-tight">
          <p className="text-[10px] font-semibold text-blue-200">
            LSTM Decoder
          </p>
          <p className="font-mono text-[8.5px] text-blue-400/80">
            512 hidden units
          </p>
        </div>

        <Connector />

        {/* Node 8: Deep Output Layer */}
        <div className="rounded-lg border border-slate-500/30 bg-slate-500/[0.06] p-1.5 text-center leading-tight">
          <p className="text-[10px] font-semibold text-slate-200">
            Deep Output Layer
          </p>
          <p className="font-mono text-[8.5px] text-slate-400/80">
            1,792 → 512 → 2,661
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

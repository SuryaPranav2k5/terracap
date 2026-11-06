"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ArchitectureCard } from "@/components/geocaption/architecture-card";
import { Cpu, Layers, Eye, Brain, Sparkles, Database } from "lucide-react";

interface ArchitectureModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ARCH_SPECS = [
  {
    icon: Cpu,
    title: "Visual Encoder",
    detail: "ConvNeXt-Tiny",
    desc: "Pretrained ImageNet-1K features capturing multi-scale aerial landforms. Frozen to preserve generalized vision representations.",
    meta: "27.8M parameters",
    color: "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
  },
  {
    icon: Layers,
    title: "Parameter-Efficient Tuning",
    detail: "LoRA Adapters (r=8, α=16)",
    desc: "Low-rank weight update matrices injected into depthwise convolutions to adapt features to remote sensing domain.",
    meta: "1.06M trainable parameters",
    color: "text-amber-400 border-amber-500/30 bg-amber-500/10",
  },
  {
    icon: Eye,
    title: "Spatial Attention Mechanism",
    detail: "Additive Bahdanau",
    desc: "Computes 8×8 dynamic alignment weights over 64 visual tokens for every generated word token.",
    meta: "64 tokens (8×8) × 768-d",
    color: "text-rose-400 border-rose-500/30 bg-rose-500/10",
  },
  {
    icon: Brain,
    title: "Decoder & Context Gating",
    detail: "LSTM + β-Gate",
    desc: "512-dim recurrent hidden state modulated by a sigmoid gating scalar that dynamically weights visual vs linguistic context.",
    meta: "512 hidden units",
    color: "text-purple-400 border-purple-500/30 bg-purple-500/10",
  },
  {
    icon: Database,
    title: "Vocabulary & Output",
    detail: "Deep Output Projection",
    desc: "Projects hidden representation, context vector, and previous token embedding into NWPU vocabulary logits.",
    meta: "2,661 unique tokens",
    color: "text-sky-400 border-sky-500/30 bg-sky-500/10",
  },
  {
    icon: Sparkles,
    title: "Inference Strategy",
    detail: "Beam Search (k=3)",
    desc: "Heuristic tree search finding the highest probability word sequences with length normalization.",
    meta: "Sub-second inference",
    color: "text-amber-300 border-amber-400/30 bg-amber-400/10",
  },
];

export function ArchitectureModal({ open, onOpenChange }: ArchitectureModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[96vw] max-w-[96vw] sm:max-w-4xl xl:max-w-5xl max-h-[92vh] overflow-y-auto border-white/10 bg-[#07090e]/95 p-4 sm:p-6 text-white backdrop-blur-2xl">
        <DialogHeader className="pb-3 border-b border-white/10">
          <DialogTitle className="text-xl font-bold flex items-center gap-2 text-white">
            <span className="font-display bg-gradient-to-r from-[#ffe49e] via-[#e5a84b] to-[#c98a2c] bg-clip-text text-transparent">
              NWPUCaptioner
            </span>
            <span>Model Architecture</span>
          </DialogTitle>
          <DialogDescription className="text-xs text-white/60">
            End-to-end vision-language pipeline: ConvNeXt-Tiny + LoRA + Additive Spatial Attention + Gated LSTM Decoder.
          </DialogDescription>
        </DialogHeader>

        {/* Two Columns: Interactive Pipeline Flow (Left) & Technical Specifications (Right) */}
        <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] gap-5 items-start mt-4">
          {/* Left Column: Visual Pipeline Flow */}
          <div className="w-full">
            <ArchitectureCard />
          </div>

          {/* Right Column: Architectural Specifications */}
          <div className="flex flex-col gap-3 min-w-0">
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3 backdrop-blur-md">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-300 mb-1">
                System Overview
              </h3>
              <p className="text-[11.5px] leading-relaxed text-white/80">
                The model architecture couples a modern hierarchical vision transformer backbone (ConvNeXt-Tiny) with parameter-efficient fine-tuning (LoRA), feeding a spatial attention LSTM decoder that selectively focuses on relevant geographic features at each word generation step.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {ARCH_SPECS.map((spec) => (
                <div
                  key={spec.title}
                  className="rounded-xl border border-white/10 bg-black/40 p-3 flex flex-col justify-between hover:border-white/20 transition-colors"
                >
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-1.5">
                        <span className={`flex h-5 w-5 items-center justify-center rounded-md border ${spec.color}`}>
                          <spec.icon className="h-3 w-3" />
                        </span>
                        <span className="text-[11.5px] font-semibold text-white">
                          {spec.title}
                        </span>
                      </div>
                    </div>
                    <p className="font-mono text-[10.5px] font-medium text-amber-200/90 mb-1">
                      {spec.detail}
                    </p>
                    <p className="text-[10px] leading-snug text-white/60">
                      {spec.desc}
                    </p>
                  </div>
                  <div className="mt-2 pt-2 border-t border-white/[0.06] flex items-center justify-between">
                    <span className="font-mono text-[9px] text-emerald-400">
                      {spec.meta}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}


"use client";

import { Upload, Layers, FileText, Eye } from "lucide-react";

  export function WorkflowLanding() {
    return (
      <div className="w-full max-w-[1440px] mx-auto px-4 sm:px-8 pb-8 pt-2">
        <div className="flex flex-col gap-2.5">
          <div>
            <p className="font-mono text-[9px] font-semibold uppercase tracking-[0.24em] text-white/50">
              WORKFLOW
            </p>
            <h2 className="font-display text-xl sm:text-[22px] font-bold text-white tracking-tight">
              From Image to Insight
            </h2>
          </div>

          {/* 4 Informational Pipeline Steps (Non-clickable workflow cards) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-center">
            {/* Step 1: Upload Image */}
              <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-[#081215]/80 p-3.5 text-left backdrop-blur-md">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400">
                  <Upload className="h-5 w-5" />
                </div>
                <div className="leading-tight min-w-0">
                    <p className="text-[13px] font-semibold text-white">
                      1 · Upload Image
                    </p>
                    <p className="text-[10px] text-white/50 truncate mt-0.5">
                      Satellite or aerial image (JPG, PNG, WebP)
                    </p>
                </div>
          </div>

            {/* Step 2: AI Encoder */}
            <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-[#081215]/80 p-3.5 text-left backdrop-blur-md">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400">
                <Layers className="h-5 w-5" />
              </div>
              <div className="leading-tight min-w-0">
                  <p className="text-[13px] font-semibold text-white">
                    2 · AI Encoder
                  </p>
                  <p className="text-[10px] text-white/50 truncate mt-0.5">
                    ConvNeXt-Tiny + LoRA Adapters
                  </p>
              </div>
        </div>

        {/* Step 3: Generate Caption */}
          <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-[#0d0d1b]/80 p-3.5 text-left backdrop-blur-md">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400">
              <FileText className="h-5 w-5" />
            </div>
            <div className="leading-tight min-w-0">
                <p className="text-[13px] font-semibold text-white">
                  3 · Generate Caption
                </p>
                <p className="text-[10px] text-white/50 truncate mt-0.5">
                  Spatial Attention LSTM Decoder
                </p>
            </div>
      </div>

          {/* Step 4: Visualize Attention */ }
      <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-[#0d0d1b]/80 p-3.5 text-left backdrop-blur-md">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-rose-500/20 text-rose-400">
          <Eye className="h-5 w-5" />
        </div>
        <div className="leading-tight min-w-0">
            <p className="text-[13px] font-semibold text-white">
              4 · Visualize Attention
            </p>
            <p className="text-[10px] text-white/50 truncate mt-0.5">
              Step-by-step spatial heatmaps
            </p>
        </div>
          </div >
        </div >
      </div >
    </div >
  );
  }

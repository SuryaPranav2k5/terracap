"use client";

import { Sparkles, ArrowRight, Box, Crosshair, FileText, Layers } from "lucide-react";

interface HeroLandingProps {
  onOpenGenerate: () => void;
  onOpenArchitecture: () => void;
}

const FEATURES = [
  {
    icon: Layers,
    title: "ConvNeXt + LoRA",
    subtitle: "Powerful visual encoder",
  },
  {
    icon: Crosshair,
    title: "Spatial Attention",
    subtitle: "Focus on what matters",
  },
  {
    icon: FileText,
    title: "Detailed Captions",
    subtitle: "Human-like descriptions",
  },
];

export function HeroLanding({ onOpenGenerate, onOpenArchitecture }: HeroLandingProps) {
  return (
    <div className="relative w-full max-w-[1440px] mx-auto px-4 sm:px-8 pt-8 sm:pt-14 pb-8 flex-1 flex flex-col justify-between">
      {/* Upper Hero Content */}
      <div className="relative z-10 max-w-[580px]">
        {/* Category Tag */}
        <p className="font-mono text-[10.5px] font-semibold uppercase tracking-[0.24em] text-white/60 mb-3">
          SATELLITE IMAGERY &nbsp;×&nbsp; DEEP LEARNING
        </p>

        {/* Headline */}
        <h1 className="font-display text-4xl sm:text-5xl lg:text-[62px] font-bold leading-[1.08] tracking-tight text-white">
          Bridging
          <br />
          <span className="bg-gradient-to-r from-[#6ee7b7] via-[#34d399] to-[#2dd4bf] bg-clip-text text-transparent">
            Earth
          </span>{" "}
          <span className="text-white/95">and</span>{" "}
          <span className="bg-gradient-to-r from-[#fef08a] via-[#fde047] to-[#eab308] bg-clip-text text-transparent">
            Language.
          </span>
        </h1>

        {/* Subtitle */}
        <p className="mt-4 text-[13px] sm:text-[14px] leading-relaxed text-white/75 max-w-[480px]">
          Generate detailed, human-like descriptions for high-resolution
          remote sensing images using deep learning and spatial attention.
        </p>

        {/* CTA Buttons */}
        <div className="mt-6 flex flex-wrap items-center gap-3">
          {/* Primary CTA: Generate Caption */}
          <button
            onClick={onOpenGenerate}
            className="group relative flex h-10 sm:h-11 items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#72f2d6] via-[#48e1c0] to-[#2cd4ab] px-5 text-[13px] sm:text-[13.5px] font-bold text-zinc-950 shadow-[0_0_24px_rgba(52,211,153,0.35)] transition-all hover:scale-[1.02] hover:brightness-105 active:scale-[0.99]"
          >
            <Sparkles className="h-4 w-4 fill-zinc-950/20 text-zinc-950" />
            <span>Generate Caption</span>
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </button>

          {/* Secondary CTA: Explore Architecture */}
          <button
            onClick={onOpenArchitecture}
            className="flex h-10 sm:h-11 items-center gap-2 rounded-xl border border-white/20 bg-black/40 px-5 text-[13px] sm:text-[13.5px] font-medium text-white/90 backdrop-blur-md transition-all hover:border-white/40 hover:bg-black/60 active:scale-[0.99]"
          >
            <Box className="h-4 w-4 text-white/70" />
            <span>Explore Architecture</span>
          </button>
        </div>

        {/* 3 Feature Badges in a single horizontal row */}
        <div className="mt-8 flex flex-wrap items-center gap-4 sm:gap-6">
          {FEATURES.map((f) => (
            <div key={f.title} className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-teal-400/40 bg-teal-400/10 text-teal-300">
                <f.icon className="h-4 w-4" strokeWidth={1.75} />
              </div>
              <div className="leading-tight">
                <p className="text-[12px] font-semibold text-white/95">
                  {f.title}
                </p>
                <p className="text-[10px] text-white/55">
                  {f.subtitle}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

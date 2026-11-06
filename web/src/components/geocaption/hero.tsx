"use client";

import Image from "next/image";
import { Box, Crosshair, FileText } from "lucide-react";

const FEATURES = [
  {
    icon: Box,
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

export function Hero() {
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/10 bg-[#07090e] shadow-xl min-h-[225px] flex items-center">
      {/* Right side Earth background image (hero-earth-bg.jpg) */}
      <div className="pointer-events-none absolute inset-y-0 right-0 w-full lg:w-[65%] select-none overflow-hidden">
        <Image
          src="/images/hero-earth-bg.jpg"
          alt="Earth orbital background with satellite imagery"
          fill
          priority
          sizes="(max-width: 1024px) 100vw, 65vw"
          className="object-cover object-[76%_24%] opacity-90 lg:opacity-100"
        />
        {/* Subtle gradient overlay to blend into the dark left panel */}
        <div className="absolute inset-0 bg-gradient-to-r from-[#07090e] via-[#07090e]/70 to-transparent w-[45%]" />
      </div>

      {/* Coordinates above the globe */}
      <div className="pointer-events-none absolute left-[46%] lg:left-[48%] top-3.5 hidden sm:block text-left font-mono text-[9.5px] tracking-wider text-amber-200/60 leading-tight">
        <div>37.6189° N</div>
        <div>122.3750° W</div>
      </div>

      {/* Callout box pointing to airport tile on the right */}
      <div className="pointer-events-none absolute right-3 lg:right-6 top-[22%] z-20 hidden md:block max-w-[195px] lg:max-w-[210px]">
        {/* Angled pointer line */}
        <div className="absolute -left-5 top-4 h-px w-5 bg-gradient-to-r from-amber-400/80 to-transparent" />
        <div className="rounded-lg border border-amber-500/30 bg-black/85 p-2 text-[10.5px] leading-snug text-white/90 shadow-2xl backdrop-blur-md">
          A large airport with multiple runways, aircraft parked at gates, and surrounding terminal buildings.
        </div>
      </div>

      {/* Bottom right annotation */}
      <div className="pointer-events-none absolute bottom-2.5 right-4 z-20 hidden sm:flex items-center gap-1.5 text-[10px] text-white/50">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]" />
        <span>High-resolution imagery.</span>
        <span className="text-white/80 font-medium">Deeper understanding.</span>
      </div>

      {/* Left Content (Text, Badge, Headline, Capsules) */}
      <div className="relative z-10 flex flex-col justify-center px-4 sm:px-6 py-4 max-w-[560px]">
        {/* Green Pill Badge */}
        <div className="mb-2 inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 w-fit">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
          <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-emerald-400">
            AI for a clearer Earth
          </span>
        </div>

        {/* Headline */}
        <h1 className="font-display text-3xl sm:text-[34px] font-bold leading-[1.08] tracking-tight text-white">
          See the Earth
          <br />
          <span className="text-white">in </span>
          <span className="bg-gradient-to-r from-[#ffe49e] via-[#e5a84b] to-[#c98a2c] bg-clip-text text-transparent italic">
            Words.
          </span>
        </h1>

        {/* Subtitle */}
        <p className="mt-2 text-[11px] sm:text-[11.5px] leading-relaxed text-white/70 max-w-[420px]">
          Generate detailed, human-like descriptions for high-resolution
          remote sensing images using deep learning and spatial attention.
        </p>

        {/* 3 Feature Capsules in ONE horizontal line */}
        <div className="mt-4 flex flex-nowrap items-center gap-2 overflow-x-auto pb-0.5">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/50 px-2.5 py-1.5 backdrop-blur-sm shrink-0"
            >
              <div className="flex h-6.5 w-6.5 shrink-0 items-center justify-center rounded-md border border-amber-400/30 bg-amber-400/10 text-amber-300">
                <f.icon className="h-3.5 w-3.5" strokeWidth={1.75} />
              </div>
              <div className="leading-none">
                <p className="text-[10px] font-semibold text-white/95 leading-none">
                  {f.title}
                </p>
                <p className="text-[8.5px] text-white/50 leading-none mt-1">
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

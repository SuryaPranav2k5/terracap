"use client";

import Image from "next/image";

/**
 * Hero visualization — High-fidelity orbital Earth render with floating satellite tiles.
 * Matches Image 1 using user-provided Image 2 (hero-earth-bg.jpg).
 */
export function EarthVisual() {
  return (
    <div className="relative mx-auto aspect-[16/10] w-full max-w-[760px] select-none">
      {/* Background orbital Earth render */}
      <div className="relative h-full w-full overflow-hidden rounded-2xl [mask-image:radial-gradient(ellipse_95%_95%_at_60%_45%,black_70%,transparent_100%)]">
        <Image
          src="/images/hero-earth-bg.jpg"
          alt="Orbital Earth with high-resolution satellite imagery tiles"
          fill
          priority
          sizes="(max-width: 1024px) 100vw, 50vw"
          className="object-contain object-right"
        />
      </div>

      {/* Coordinates at top */}
      <div className="absolute right-[44%] top-[12%] text-right font-mono text-[10px] tracking-wider text-amber-200/60 sm:text-[11px]">
        <div>37.6189° N</div>
        <div>122.3750° W</div>
      </div>

      {/* Callout box next to airport tile */}
      <div className="absolute right-[2%] top-[18%] sm:right-[4%] sm:top-[22%] z-20 max-w-[210px] sm:max-w-[240px]">
        {/* Connector line */}
        <div className="absolute -left-6 top-6 h-px w-6 bg-gradient-to-r from-amber-400/80 to-transparent" />
        <div className="rounded-xl border border-amber-500/30 bg-black/80 p-3 text-[11.5px] sm:text-[12px] leading-snug text-white/90 shadow-2xl backdrop-blur-md">
          A large airport with multiple runways, aircraft parked at gates, and surrounding terminal buildings.
        </div>
      </div>

      {/* Bottom right annotation */}
      <div className="absolute bottom-[4%] right-[2%] text-right text-[11.5px] sm:text-[12.5px] text-white/50 leading-tight">
        <div>High-resolution imagery.</div>
        <div className="text-white/70">Deeper understanding.</div>
      </div>
    </div>
  );
}

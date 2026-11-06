"use client";

import { Globe } from "lucide-react";
import { GeoCaptionLogo } from "@/components/geocaption/logo";
import { scrollToSection } from "@/hooks/use-scroll-spy";

const LINKS = [
  { id: "generate", label: "Generate" },
  { id: "architecture", label: "Architecture" },
  { id: "dataset", label: "Dataset" },
  { id: "results", label: "Results" },
  { id: "gallery", label: "Gallery" },
  { id: "about", label: "About" },
];

export function Footer() {
  return (
    <footer className="mt-auto border-t border-gc-border bg-gc-bg/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1400px] flex-col items-start justify-between gap-6 px-4 py-9 sm:px-6 md:flex-row md:items-center lg:px-10">
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center gap-2.5">
            <GeoCaptionLogo size={26} />
            <span className="font-display text-lg font-semibold text-gc-gold">
              TerraCap
            </span>
          </div>
          <p className="flex items-center gap-1.5 text-[12px] text-gc-faint">
            <Globe className="h-3.5 w-3.5 text-gc-gold/70" />
            Bridging Satellite Imagery and Human Language.
          </p>
          <p className="font-mono text-[11px] text-gc-faint">
            © {new Date().getFullYear()} TerraCap · Live inference via a
            vision-language model
          </p>
        </div>

        <nav
          aria-label="Footer"
          className="flex flex-wrap items-center gap-x-6 gap-y-2"
        >
          {LINKS.map((l) => (
            <button
              key={l.id}
              onClick={() => scrollToSection(l.id)}
              className="text-[13px] text-gc-dim transition-colors hover:text-gc-gold"
            >
              {l.label}
            </button>
          ))}
        </nav>
      </div>
    </footer>
  );
}

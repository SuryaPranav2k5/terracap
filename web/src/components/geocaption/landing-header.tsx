"use client";

import Link from "next/link";
import Image from "next/image";
import { Github } from "lucide-react";

export function LandingHeader() {
  return (
    <header className="w-full z-50 pt-4 px-4 sm:px-8 max-w-[1440px] mx-auto flex items-center justify-between">
      {/* Left: Brand Logo & Wordmark */}
      <div className="flex items-center gap-2.5">
        <div className="relative h-7 w-7 shrink-0">
          <Image
            src="/extracted-logo.png"
            alt="TerraCap Logo"
            fill
            priority
            className="object-contain"
          />
        </div>
        <span className="font-display text-[20px] font-bold tracking-tight text-white">
          TerraCap
        </span>
      </div>

      {/* Right: GitHub Pill Button */}
      <div className="flex items-center">
        <Link
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-8 items-center gap-2 rounded-full border border-white/20 bg-black/50 px-3.5 text-[12px] font-medium text-white/90 backdrop-blur-md transition-all hover:border-white/40 hover:bg-black/70"
        >
          <Github className="h-3.5 w-3.5 text-white" />
          GitHub
        </Link>
      </div>
    </header>
  );
}

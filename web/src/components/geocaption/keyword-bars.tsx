"use client";

import type { Keyword } from "@/lib/types";

export function KeywordBars({ keywords }: { keywords?: Keyword[] }) {
  if (!keywords || keywords.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-3 text-center">
        <p className="text-[11.5px] text-white/40 italic">
          Keywords will appear after caption generation.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col justify-center gap-1.5">
      {keywords.slice(0, 5).map((kw, i) => (
        <div key={`${kw.word}-${i}`} className="flex items-center gap-2.5">
          <span className="w-[72px] shrink-0 truncate text-left font-mono text-[11px] font-medium text-white/85">
            {kw.word}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-400 to-cyan-400 shadow-[0_0_6px_rgba(45,212,191,0.5)]"
              style={{
                width: `${Math.round(kw.score * 100)}%`,
                transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)",
                transitionDelay: `${i * 70}ms`,
              }}
            />
          </div>
          <span className="w-8 shrink-0 font-mono text-[10.5px] text-white/60 text-right">
            {kw.score.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

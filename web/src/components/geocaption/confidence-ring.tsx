"use client";

import { useEffect, useState } from "react";

export function ConfidenceRing({
  confidence,
  size = 78,
  stroke = 7,
}: {
  confidence: number;
  size?: number;
  stroke?: number;
}) {
  const [progress, setProgress] = useState(0);
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const isSet = confidence > 0;
  const pct = isSet ? Math.round(confidence * 100) : 0;

  useEffect(() => {
    const t = setTimeout(() => setProgress(isSet ? confidence : 0), 120);
    return () => clearTimeout(t);
  }, [confidence, isSet]);

  return (
    <div
      className="relative shrink-0 flex items-center justify-center"
      role="img"
      aria-label={isSet ? `Model confidence ${pct}%` : "No confidence data"}
    >
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#34d399" />
            <stop offset="100%" stopColor="#22d3ee" />
          </linearGradient>
        </defs>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={stroke}
        />
        {isSet && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="url(#ringGrad)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - progress)}
            style={{
              transition: "stroke-dashoffset 1.1s cubic-bezier(0.4,0,0.2,1)",
              filter: "drop-shadow(0 0 6px rgba(45, 212, 191, 0.4))",
            }}
          />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-[18px] sm:text-[20px] font-bold leading-none text-white tracking-tight">
          {isSet ? `${pct}%` : "--"}
        </span>
      </div>
    </div>
  );
}

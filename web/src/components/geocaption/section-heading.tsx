"use client";

import type { LucideIcon } from "lucide-react";

export function SectionHeading({
  icon: Icon,
  overline,
  title,
  description,
}: {
  icon: LucideIcon;
  overline?: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="mb-8">
      {overline && (
        <p className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-gc-gold">
          <Icon className="h-4 w-4" />
          {overline}
        </p>
      )}
      <h2 className="font-display text-3xl font-bold tracking-tight text-gc-text sm:text-4xl">
        {title}
      </h2>
      {description && (
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-gc-dim">
          {description}
        </p>
      )}
    </div>
  );
}

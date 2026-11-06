"use client";

import { useEffect, useState } from "react";

/**
 * Tracks which page section is currently in view (scroll-spy).
 */
export function useScrollSpy(ids: string[], offset = 0.4): string {
  const [active, setActive] = useState(ids[0] ?? "");

  useEffect(() => {
    const handler = () => {
      const viewportMid = window.innerHeight * offset;
      let current = ids[0] ?? "";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        if (rect.top <= viewportMid) {
          current = id;
        }
      }
      setActive(current);
    };
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, [ids, offset]);

  return active;
}

export function scrollToSection(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const top = el.getBoundingClientRect().top + window.scrollY - 76;
  window.scrollTo({ top, behavior: "smooth" });
}

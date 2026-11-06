"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Reveals text progressively (typewriter effect) whenever the text changes.
 * Driven by requestAnimationFrame so state updates happen outside the effect
 * body. Returns the visible slice and whether the animation finished.
 */
export function useTypewriter(
  text: string,
  charsPerFrame = 2
): { displayed: string; done: boolean } {
  const [count, setCount] = useState(0);
  const prevText = useRef<string | null>(null);

  useEffect(() => {
    if (prevText.current === text) return;
    prevText.current = text;
    if (!text) return;

    let raf = 0;
    let c = 0;
    const tick = () => {
      c = Math.min(text.length, c + charsPerFrame);
      setCount(c);
      if (c < text.length) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, charsPerFrame]);

  const effective = Math.min(count, text.length);
  return {
    displayed: text.slice(0, effective),
    done: effective >= text.length,
  };
}

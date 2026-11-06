"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Eye, Play, Pause } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { useGenStore } from "@/lib/store";
import {
  computeSaliency,
  renderAttention,
  renderRealAttentionGrid,
  type SaliencyMap,
} from "@/lib/saliency";

export function AttentionCard() {
  const image = useGenStore((s) => s.image);
  const result = useGenStore((s) => s.result);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const salRef = useRef<SaliencyMap | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef(0);

  const [ready, setReady] = useState(false);
  const [version, setVersion] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);

  const attentionSteps = result?.attentionSteps;
  const hasRealAttention =
    Array.isArray(attentionSteps) && attentionSteps.length > 0;
  const totalSteps = hasRealAttention ? attentionSteps.length : 12;

  const currentToken =
    hasRealAttention && attentionSteps[step]
      ? attentionSteps[step].token
      : null;

  const src = image?.dataUrl ?? result?.imageSrc ?? null;

  // Reset step when result changes
  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [result?.id]);

  useEffect(() => {
    if (!src || typeof window === "undefined") return;
    let cancelled = false;
    const img = new window.Image();
    if (src.startsWith("http")) {
      img.crossOrigin = "anonymous";
    }
    img.onload = () => {
      if (cancelled) return;
      imgRef.current = img;
      salRef.current = computeSaliency(img);
      setReady(true);
      setVersion((v) => v + 1);
    };
    img.onerror = () => {
      if (!cancelled) setReady(false);
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  const draw = useCallback(
    (currentStep: number) => {
      const canvas = canvasRef.current;
      if (!canvas || !imgRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(2, Math.round(rect.width * dpr));
      const h = Math.max(2, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      if (hasRealAttention && attentionSteps[currentStep]?.grid) {
        renderRealAttentionGrid(
          canvas,
          imgRef.current,
          attentionSteps[currentStep].grid
        );
      } else if (salRef.current) {
        renderAttention(canvas, imgRef.current, salRef.current, {
          step: currentStep,
          totalSteps,
        });
      }
    },
    [hasRealAttention, attentionSteps, totalSteps]
  );

  useEffect(() => {
    if (ready) draw(step);
  }, [ready, version, step, draw]);

  useEffect(() => {
    const onResize = () => draw(step);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw, step]);

  useEffect(() => {
    if (!playing || !ready) return;
    const tick = (ts: number) => {
      if (ts - lastTickRef.current >= 450) {
        lastTickRef.current = ts;
        setStep((s) => (s + 1) % totalSteps);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, ready, totalSteps]);

  return (
      <Card className="flex flex-col justify-between rounded-2xl border border-white/10 bg-[#07090e]/90 p-5 sm:p-6 backdrop-blur-md shadow-md min-h-[360px]">
        <CardHeader className="p-0 pb-3">
          <div className="flex items-center justify-between gap-2 min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-amber-500/15 text-amber-400">
                <Eye className="h-4 w-4" strokeWidth={2} />
              </span>
                <CardTitle className="text-[16px] font-semibold text-white truncate">
                  Spatial Attention Heatmap
                </CardTitle>
            </div>
            {currentToken && (
                <span className="font-mono text-[12px] text-amber-300 font-bold px-3 py-1 rounded-md bg-amber-500/15 border border-amber-500/30 shrink-0 truncate max-w-[160px] shadow-sm">
                  &ldquo;{currentToken}&rdquo;
                </span>
          )}
              </div>
      </CardHeader>

            <CardContent className="flex flex-1 flex-col justify-between p-0 gap-3.5">
              {/* Full-width Canvas + Jet Legend (Enlarged and High Resolution) */}
              <div className="flex items-center gap-4 h-[250px] sm:h-[280px]">
                <div className="relative flex-1 h-full overflow-hidden rounded-xl border border-white/10 bg-black flex items-center justify-center shadow-inner">
                  {src ? (
                    <canvas ref={canvasRef} className="h-full w-full object-contain" />
                  ) : (
                      <p className="p-4 text-center text-white/35 text-[12px] italic">
                        Spatial attention heatmaps will visualize token-level alignment after generating a caption.
                      </p>
            )}
                    </div>

          {/* High / Low Attention Legend */}
                    <div className="flex flex-col items-center justify-between h-full py-2 text-[10px] font-semibold text-white/60 leading-tight text-center shrink-0 w-[44px]">
                      <span className="text-rose-400 font-bold">High</span>
                      <div className={`w-2.5 flex-1 my-2 rounded-full ${src ? "bg-gradient-to-b from-[#e11d48] via-[#eab308] via-[#10b981] to-[#2563eb] shadow-[0_0_10px_rgba(244,63,94,0.5)]" : "bg-white/10"}`} />
                      <span className="text-sky-400 font-bold">Low</span>
                    </div>
                  </div>

                  {/* Playback Controls */}
                    <div className="flex items-center gap-3.5 pt-1">
                      <button
                        onClick={() => setPlaying(!playing)}
                        disabled={!src || !hasRealAttention}
                        aria-label={playing ? "Pause" : "Play"}
                        className="flex h-8.5 w-8.5 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-amber-400 text-black transition-transform hover:scale-105 disabled:opacity-30 disabled:cursor-not-allowed shadow-md"
                      >
                        {playing ? (
              <Pause className="h-4 w-4 fill-black" />
                        ) : (
              <Play className="h-4 w-4 fill-black ml-0.5" />
                        )}
                      </button>
                        <div className="flex-1 min-w-[100px]">
                          <Slider
                            value={[step]}
                            max={Math.max(0, totalSteps - 1)}
                            step={1}
                            disabled={!src || !hasRealAttention}
                            onValueChange={([val]) => {
                              setPlaying(false);
                              setStep(val);
                            }}
                            className="py-1 disabled:opacity-30"
                          />
                        </div>
                          <span className="font-mono text-[12px] font-bold text-white/80 shrink-0 px-2.5 py-1 rounded-md bg-white/5 border border-white/10">
                            {src && hasRealAttention ? `${step + 1} / ${totalSteps}` : "- / -"}
                          </span>
                      </div>
                    </CardContent>
                  </Card>
                  );
}

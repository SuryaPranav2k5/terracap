"use client";

import { motion } from "framer-motion";
import { Info, Upload, ScanEye, MessageSquareText, Sparkles } from "lucide-react";
import { SectionHeading } from "@/components/geocaption/section-heading";
import { scrollToSection } from "@/hooks/use-scroll-spy";

const STEPS = [
  {
    icon: Upload,
    title: "1 · Upload",
    body: "Drop any satellite or aerial image — the client prepares it for inference locally.",
  },
  {
    icon: ScanEye,
    title: "2 · Encode",
    body: "Visual features are extracted and salient regions located via spatial attention.",
  },
  {
    icon: MessageSquareText,
    title: "3 · Describe",
    body: "The decoder emits a fluent, detailed sentence grounded in what is actually visible.",
  },
];

const STACK = [
  "Next.js 16",
  "React 19",
  "TypeScript",
  "Tailwind CSS 4",
  "shadcn/ui",
  "PyTorch (ConvNeXt-Tiny + LoRA)",
  "Spatial Attention LSTM Decoder",
  "Canvas Attention Maps",
  "Zustand",
  "Framer Motion",
];

export function AboutSection() {
  return (
    <section id="about" className="relative py-16 sm:py-20">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10">
        <SectionHeading
          icon={Info}
          overline="About"
          title="Why we caption the Earth"
        />

        <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
          <div className="flex flex-col gap-6">
            <p className="max-w-2xl text-[15.5px] leading-relaxed text-gc-dim">
              Satellites capture more imagery every day than any team of
              analysts can describe.{" "}
              <span className="text-gc-text">GeoCaption</span> turns those
              pixels into plain language — detailed, human-like descriptions
              that make remote sensing archives searchable, accessible and
              useful to everyone, not just geospatial experts.
            </p>
            <p className="max-w-2xl text-[15.5px] leading-relaxed text-gc-dim">
              This demo runs the full loop live: your image is analyzed by a
              hosted vision-language model, spatial attention is estimated in
              your browser, and every result is archived to a real database.
              No canned responses, no placeholder data.
            </p>

            <div className="grid gap-3 sm:grid-cols-3">
              {STEPS.map((s, i) => (
                <motion.div
                  key={s.title}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ delay: i * 0.08, duration: 0.45 }}
                  className="glass-card card-hover rounded-xl p-4"
                >
                  <s.icon className="h-5 w-5 text-gc-gold" strokeWidth={1.75} />
                  <p className="mt-2.5 text-[13.5px] font-semibold text-gc-text">
                    {s.title}
                  </p>
                  <p className="mt-1 text-[12.5px] leading-relaxed text-gc-dim">
                    {s.body}
                  </p>
                </motion.div>
              ))}
            </div>

            <button
              onClick={() => scrollToSection("generate")}
              className="btn-gold flex h-12 w-fit items-center gap-2 self-start rounded-xl px-6 text-[15px] font-semibold"
            >
              <Sparkles className="h-4.5 w-4.5" />
              Try it with your own image
            </button>
          </div>

          <div className="glass-card flex flex-col gap-5 rounded-2xl p-6">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gc-faint">
                Built with
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {STACK.map((t) => (
                  <span
                    key={t}
                    className="rounded-full border border-gc-border bg-gc-panel px-3 py-1.5 text-[12.5px] text-gc-dim"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div className="border-t border-gc-border pt-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gc-faint">
                Methodology
              </p>
              <ul className="mt-3 flex flex-col gap-2.5 text-[13px] leading-relaxed text-gc-dim">
                <li className="flex gap-2.5">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-gc-gold" />
                  Captions, keyword scores and confidence values are produced
                  by the live model for each image — nothing is pre-written.
                </li>
                <li className="flex gap-2.5">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-gc-gold" />
                  Attention heatmaps are computed from your image with a
                  multi-scale center-surround saliency model on canvas.
                </li>
                <li className="flex gap-2.5">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-gc-gold" />
                  Real-time PyTorch inference runs on your images with full spatial attention heatmaps.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

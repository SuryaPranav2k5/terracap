"use client";

import { motion } from "framer-motion";
import {
  Layers,
  Snowflake,
  Plug,
  Crosshair,
  SlidersHorizontal,
  Braces,
  Info,
} from "lucide-react";
import { SectionHeading } from "@/components/geocaption/section-heading";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const TRAINING_ROWS = [
  ["Optimizer", "AdamW (β₁ 0.9, β₂ 0.999)"],
  ["Learning rate", "3e-4 (LoRA) · 1e-5 (decoder)"],
  ["Batch size", "64 images"],
  ["Epochs", "30 with early stopping"],
  ["Loss", "Cross-entropy + label smoothing (0.1)"],
  ["Hardware", "NVIDIA A100 40GB"],
  ["Framework", "PyTorch 2.1"],
];

const COMPONENTS = [
  {
    icon: Snowflake,
    title: "Frozen encoder",
    body: "A ConvNeXt-Tiny backbone (27.8M parameters) extracts robust visual features without being fine-tuned — keeping the representation general while slashing trainable compute.",
    accent: "text-emerald-500",
  },
  {
    icon: Plug,
    title: "LoRA adapters",
    body: "Low-rank adaptation matrices (1.06M trainable parameters, ~3.7% of the backbone) inject task-specific tuning into attention and MLP projections at a fraction of full fine-tuning cost.",
    accent: "text-gc-gold",
  },
  {
    icon: Crosshair,
    title: "Attention & gating",
    body: "Additive (Bahdanau) spatial attention weighs the 8 × 8 feature tokens at every decoding step; a learned β-gate blends attended features with the recurrent context.",
    accent: "text-violet-500",
  },
  {
    icon: Braces,
    title: "LSTM decoder",
    body: "A 512-unit LSTM unrolls across timesteps with a deep output layer (1,792 → 512 → 2,661) projecting onto the caption vocabulary, trained with scheduled sampling.",
    accent: "text-blue-500",
  },
];

export function PipelineSection() {
  return (
    <section id="architecture" className="relative py-16 sm:py-20">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10">
        <SectionHeading
          icon={Layers}
          overline="Architecture"
          title="How a picture becomes a paragraph"
          description="GeoCaption couples a parameter-efficient vision encoder with an attention-gated recurrent decoder — every module below is part of the captioning pipeline you just used."
        />

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {COMPONENTS.map((c, i) => (
            <motion.div
              key={c.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: i * 0.08, duration: 0.5 }}
            >
              <Card className="glass-card card-hover h-full gap-3 rounded-2xl border-gc-border py-5">
                <CardHeader className="px-5">
                  <CardTitle className="flex items-center gap-2.5 text-[15.5px] font-semibold text-gc-text">
                    <c.icon className={`h-5 w-5 ${c.accent}`} strokeWidth={1.75} />
                    {c.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-5">
                  <p className="text-[13.5px] leading-relaxed text-gc-dim">
                    {c.body}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
          <Card className="glass-card rounded-2xl border-gc-border py-5">
            <CardHeader className="px-5">
              <CardTitle className="flex items-center gap-2.5 text-[15.5px] font-semibold text-gc-text">
                <SlidersHorizontal className="h-5 w-5 text-gc-gold" />
                Training configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="px-5">
              <dl className="grid gap-x-8 gap-y-0 sm:grid-cols-2">
                {TRAINING_ROWS.map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-baseline justify-between gap-4 border-b border-gc-border py-2.5 last:border-b-0 sm:last:border-b sm:[&:nth-last-child(2)]:border-b-0"
                  >
                    <dt className="text-[13px] text-gc-faint">{k}</dt>
                    <dd className="text-right font-mono text-[12.5px] text-gc-text">
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-gc-gold/30 bg-gc-gold/5 py-5">
            <CardHeader className="px-5">
              <CardTitle className="flex items-center gap-2.5 text-[15.5px] font-semibold text-gc-gold">
                <Info className="h-5 w-5" />
                Live inference notice
              </CardTitle>
            </CardHeader>
            <CardContent className="px-5">
              <p className="text-[13.5px] leading-relaxed text-gc-dim">
                The architecture above describes the GeoCaption captioning
                system design. The demo you are using performs{" "}
                <span className="text-gc-text">
                  live inference with a hosted vision-language model
                </span>{" "}
                over your uploaded imagery, and the attention panel computes
                real multi-scale saliency maps in your browser. Every caption,
                score and gallery entry on this page was generated on the fly —
                nothing is canned.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}

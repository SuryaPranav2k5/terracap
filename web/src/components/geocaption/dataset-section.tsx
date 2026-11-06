"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { Database, ArrowUpRight } from "lucide-react";
import { SectionHeading } from "@/components/geocaption/section-heading";
import { Card, CardContent } from "@/components/ui/card";
import { useGenStore } from "@/lib/store";
import { scrollToSection } from "@/hooks/use-scroll-spy";
import type { GenerationItem } from "@/lib/types";

const DATASET_FACTS = [
  { value: "10,921", label: "RS images" },
  { value: "21", label: "Land-use classes" },
  { value: "54,704", label: "Reference captions" },
  { value: "256²", label: "Pixels per image" },
];

const CLASS_ICONS: Record<string, string> = {
  Airport: "✈",
  Harbor: "⚓",
  Stadium: "◉",
  Farmland: "▤",
  Industrial: "▦",
  Residential: "▩",
  Overpass: "⤬",
  Beach: "≈",
};

export function DatasetSection() {
  const loadGeneration = useGenStore((s) => s.loadGeneration);
  const [items, setItems] = useState<GenerationItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/generations?source=reference&limit=100")
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => {
        if (!cancelled) {
          setItems(data.items ?? []);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section id="dataset" className="relative py-16 sm:py-20">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10">
        <SectionHeading
          icon={Database}
          overline="Dataset"
          title="Trained to read the ground"
          description="GeoCaption's reference corpus is built on RSICD — the Remote Sensing Image Captioning Dataset — with 21 land-use classes captured at 0.3–3 m GSD. Explore the reference classes below; each tile loads its live model output."
        />

        {/* Facts strip */}
        <div className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {DATASET_FACTS.map((f) => (
            <div
              key={f.label}
              className="glass-card card-hover rounded-xl px-5 py-4"
            >
              <p className="font-mono text-2xl font-bold text-gc-gold">
                {f.value}
              </p>
              <p className="mt-0.5 text-[11.5px] font-semibold uppercase tracking-[0.12em] text-gc-faint">
                {f.label}
              </p>
            </div>
          ))}
        </div>

        {/* Class explorer */}
        {loading ? (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="shimmer aspect-[4/3] rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
            {items.map((gen, i) => {
              const label = gen.label ?? gen.imageName.replace(/\.[a-z]+$/i, "");
              return (
                <motion.button
                  key={gen.id}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ delay: (i % 4) * 0.06, duration: 0.45 }}
                  onClick={() => {
                    loadGeneration(gen);
                    scrollToSection("generate");
                  }}
                  className="group relative overflow-hidden rounded-xl border border-gc-border text-left outline-none transition-colors hover:border-gc-gold/60 focus-visible:ring-2 focus-visible:ring-ring"
                  title="Open this sample in the captioning panel"
                >
                  <div className="relative aspect-[4/3]">
                    <Image
                      src={gen.imageSrc}
                      alt={`${label} class sample`}
                      fill
                      unoptimized={gen.imageSrc.startsWith("data:")}
                      sizes="(max-width: 640px) 45vw, (max-width: 1280px) 30vw, 22vw"
                      className="object-cover transition-transform duration-500 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/10 to-transparent" />
                    <span className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-md bg-black/50 text-[12px] text-gc-gold opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
                      <ArrowUpRight className="h-3.5 w-3.5" />
                    </span>
                    <div className="absolute inset-x-0 bottom-0 p-3">
                      <p className="flex items-center gap-2 text-[14px] font-semibold text-white">
                        <span className="text-gc-gold">
                          {CLASS_ICONS[label] ?? "▣"}
                        </span>
                        {label}
                      </p>
                      <p className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug text-white/75">
                        {gen.caption}
                      </p>
                    </div>
                  </div>
                </motion.button>
              );
            })}
          </div>
        )}

        <Card className="glass-card mt-6 rounded-2xl border-gc-border py-4">
          <CardContent className="px-5">
            <p className="text-[12.5px] leading-relaxed text-gc-faint">
              Clicking a class tile loads that reference image and its live
              model caption into the Generate panel — the same pipeline your
              own uploads run through.
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}

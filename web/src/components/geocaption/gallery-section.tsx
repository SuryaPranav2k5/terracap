"use client";

import { useCallback, useEffect, useState } from "react";
import Image from "next/image";
import { LayoutGrid, Loader2 } from "lucide-react";
import { SectionHeading } from "@/components/geocaption/section-heading";
import { GenerationDetailDialog } from "@/components/geocaption/generation-detail-dialog";
import { useGenStore } from "@/lib/store";
import type { GenerationItem } from "@/lib/types";
import { timeAgo } from "@/lib/format";

const PAGE_SIZE = 12;

export function GallerySection() {
  const galleryVersion = useGenStore((s) => s.galleryVersion);
  const [items, setItems] = useState<GenerationItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selected, setSelected] = useState<GenerationItem | null>(null);

  const load = useCallback(
    async (offset: number) => {
      const res = await fetch(
        `/api/generations?limit=${PAGE_SIZE}&offset=${offset}`
      );
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    []
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load(0)
      .then((data) => {
        if (!cancelled) {
          setItems(data.items ?? []);
          setTotal(data.total ?? 0);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [load, galleryVersion]);

  const loadMore = async () => {
    if (items.length >= total) return;
    setLoadingMore(true);
    try {
      const data = await load(items.length);
      setItems((prev) => {
        const seen = new Set(prev.map((g) => g.id));
        return [...prev, ...(data.items ?? []).filter((g: GenerationItem) => !seen.has(g.id))];
      });
      setTotal(data.total ?? 0);
    } catch {
      /* ignore */
    } finally {
      setLoadingMore(false);
    }
  };

  const refresh = useCallback(() => {
    void load(0).then((data) => {
      setItems(data.items ?? []);
      setTotal(data.total ?? 0);
    });
  }, [load]);

  return (
    <section id="gallery" className="relative py-16 sm:py-20">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10">
        <SectionHeading
          icon={LayoutGrid}
          overline="Gallery"
          title="Every caption, live from the model"
          description="A running archive of real generations — reference samples and every image captioned in this session. Click any card for the full result."
        />

        {loading ? (
          <div className="columns-2 gap-4 md:columns-3 xl:columns-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div
                key={i}
                className="shimmer mb-4 break-inside-avoid rounded-xl"
                style={{ height: `${160 + (i % 3) * 50}px` }}
              />
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="glass-card flex flex-col items-center justify-center gap-3 rounded-2xl px-6 py-16 text-center">
            <LayoutGrid className="h-10 w-10 text-gc-faint" strokeWidth={1.5} />
            <p className="text-[14px] text-gc-dim">
              The gallery is empty — generate your first caption above and it
              will be archived here.
            </p>
          </div>
        ) : (
          <>
            <div className="columns-2 gap-4 md:columns-3 xl:columns-4">
              {items.map((gen) => (
                <button
                  key={gen.id}
                  onClick={() => setSelected(gen)}
                  className="group relative mb-4 block w-full break-inside-avoid overflow-hidden rounded-xl border border-gc-border text-left outline-none transition-colors hover:border-gc-gold/60 focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <div className="relative">
                    <Image
                      src={gen.imageSrc}
                      alt={gen.caption.slice(0, 80)}
                      width={640}
                      height={480}
                      unoptimized={gen.imageSrc.startsWith("data:")}
                      sizes="(max-width: 768px) 45vw, (max-width: 1280px) 30vw, 22vw"
                      className="h-auto w-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/25 opacity-80 transition-opacity group-hover:opacity-100" />
                    <span className="absolute left-2 top-2 rounded-full bg-black/55 px-2 py-0.5 font-mono text-[10px] text-gc-gold backdrop-blur-sm">
                      {Math.round(gen.confidence * 100)}%
                    </span>
                    <span className="absolute right-2 top-2 rounded-full bg-black/55 px-2 py-0.5 text-[10px] uppercase tracking-wider text-white/85 backdrop-blur-sm">
                      {gen.source === "upload" ? "Live" : "Reference"}
                    </span>
                    <div className="absolute inset-x-0 bottom-0 p-3">
                      <p className="line-clamp-2 text-[12.5px] leading-snug text-white/95">
                        {gen.caption}
                      </p>
                      <p className="mt-1 font-mono text-[10px] text-white/55">
                        {timeAgo(gen.createdAt)}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {items.length < total && (
              <div className="mt-2 flex justify-center">
                <button
                  onClick={() => void loadMore()}
                  disabled={loadingMore}
                  className="flex h-11 items-center gap-2 rounded-full border border-gc-border-strong px-6 text-[14px] font-medium text-gc-dim transition-colors hover:border-gc-gold/60 hover:text-gc-gold disabled:opacity-50"
                >
                  {loadingMore && <Loader2 className="h-4 w-4 animate-spin" />}
                  Load more ({items.length} / {total})
                </button>
              </div>
            )}
          </>
        )}
      </div>

      <GenerationDetailDialog
        generation={selected}
        open={Boolean(selected)}
        onOpenChange={(o) => !o && setSelected(null)}
        onDeleted={refresh}
      />
    </section>
  );
}

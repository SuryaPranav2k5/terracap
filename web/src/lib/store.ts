"use client";

import { create } from "zustand";
import type {
  GenerationItem,
  PickedImage,
} from "@/lib/types";
import { REFERENCE_GENERATIONS } from "@/lib/reference-data";
import { toast } from "sonner";

export type GenStatus = "idle" | "generating" | "done" | "error";

interface GenState {
  image: PickedImage | null;
  result: GenerationItem | null;
  status: GenStatus;
  error: string | null;
  galleryVersion: number;
  initialized: boolean;
  setImage: (img: PickedImage) => void;
  clearImage: () => void;
  generate: () => Promise<void>;
  loadInitial: () => Promise<void>;
  loadNext: () => Promise<void>;
  loadGeneration: (gen: GenerationItem) => void;
}

export const useGenStore = create<GenState>((set, get) => ({
  image: null,
  result: null,
  status: "idle",
  error: null,
  galleryVersion: 0,
  initialized: false,

  setImage: (img) => set({ image: img, result: null, status: "idle", error: null }),

  clearImage: () => set({ image: null, result: null, status: "idle", error: null }),

  generate: async () => {
    const { image } = get();
    if (!image) return;
    set({ status: "generating", error: null });
    try {
      const res = await fetch("/api/caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: image.dataUrl, imageName: image.name }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      const data: GenerationItem = await res.json();
      set({
        result: { ...data, imageSrc: image.dataUrl },
        status: "done",
        galleryVersion: get().galleryVersion + 1,
      });
      toast.success(
        `Caption generated in ${(data.genTimeMs / 1000).toFixed(1)}s`
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to generate caption";
      set({ status: "error", error: message });
      toast.error(message);
    }
  },

  loadInitial: async () => {
    // Keep clean initial state (no mockup pre-loaded)
    set({ initialized: true });
  },

  loadNext: async () => {
    try {
      const res = await fetch("/api/generations?limit=100");
      let items: GenerationItem[] = REFERENCE_GENERATIONS;
      if (res.ok) {
        const data = await res.json();
        if (data.items && data.items.length > 0) {
          items = data.items;
        }
      }
      const current = get().result;
      const idx = current ? items.findIndex((g) => g.id === current.id) : -1;
      const next = items[(idx + 1) % items.length];
      set({
        result: next,
        image: {
          dataUrl: next.imageSrc,
          name: next.imageName,
          sizeLabel: next.source === "reference" ? "Reference" : "Upload",
        },
        status: "done",
        error: null,
      });
    } catch {
      toast.error("Could not load the next generation.");
    }
  },

  loadGeneration: (gen) => {
    set({
      result: gen,
      image: {
        dataUrl: gen.imageSrc,
        name: gen.imageName,
        sizeLabel: gen.source === "reference" ? "Reference" : "Upload",
      },
      status: "done",
      error: null,
    });
  },
}));

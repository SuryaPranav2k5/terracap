import type { GenerationItem } from "@/lib/types";
import referenceDataJson from "./reference-data.json";

export const REFERENCE_GENERATIONS: GenerationItem[] = referenceDataJson as GenerationItem[];

export const DEFAULT_SAMPLE_ITEM: GenerationItem =
  REFERENCE_GENERATIONS.find((g) => g.imageName === "harbor.png") ??
  REFERENCE_GENERATIONS[0];


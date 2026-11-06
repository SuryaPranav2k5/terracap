export interface Keyword {
  word: string;
  score: number;
}

export interface AttentionStep {
  step: number;
  token: string;
  grid: number[][]; // 8x8 matrix
}

export interface GenerationItem {
  id: string;
  imageSrc: string;
  imageName: string;
  label: string | null;
  caption: string;
  keywords: Keyword[];
  confidence: number;
  genTimeMs: number;
  source: "reference" | "upload";
  createdAt: string;
  tokens?: string[];
  attentionSteps?: AttentionStep[];
}

export type CaptionResponse = GenerationItem;

export interface PickedImage {
  dataUrl: string;
  name: string;
  sizeLabel: string;
}

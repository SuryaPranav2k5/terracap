import type { GenerationItem } from "@/lib/types";

// In-memory store for session-based user uploads
const sessionUploads: GenerationItem[] = [];

export function getSessionUploads(): GenerationItem[] {
  return sessionUploads;
}

export function addSessionUpload(record: GenerationItem): void {
  sessionUploads.unshift(record);
  if (sessionUploads.length > 50) {
    sessionUploads.pop();
  }
}


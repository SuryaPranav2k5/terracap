"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { GenerateCard } from "@/components/geocaption/generate-card";
import { CaptionCard } from "@/components/geocaption/caption-card";
import { AttentionCard } from "@/components/geocaption/attention-card";

interface WorkspaceModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function WorkspaceModal({ open, onOpenChange }: WorkspaceModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[96vw] max-w-[96vw] sm:max-w-6xl xl:max-w-7xl max-h-[95vh] overflow-y-auto border-white/10 bg-[#07090e]/95 p-5 sm:p-7 text-white backdrop-blur-2xl">
        <DialogHeader className="pb-4 border-b border-white/10">
          <DialogTitle className="text-2xl font-bold flex items-center gap-2.5 text-white">
            <span className="font-display bg-gradient-to-r from-[#6ee7b7] via-[#34d399] to-[#2dd4bf] bg-clip-text text-transparent">
              TerraCap
            </span>
            <span>Interactive Studio</span>
          </DialogTitle>
          <DialogDescription className="text-sm text-white/70 mt-1">
            Real remote sensing vision-language model inference and spatial attention visualization.
          </DialogDescription>
        </DialogHeader>

        {/* Studio Workspace Layout */}
        <div className="flex flex-col gap-5 mt-4">
          {/* Row 1: Generate Caption + Generated Caption */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
            <GenerateCard />
            <CaptionCard />
          </div>

            {/* Row 2: Full-width Spatial Attention Heatmap */}
            <div className="w-full">
              <AttentionCard />
            </div>
          </div>
      </DialogContent>
    </Dialog>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CloudUpload } from "lucide-react";
import { toast } from "sonner";
import { useGenStore } from "@/lib/store";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;

/** Downscales large images client-side so payloads stay light. */
function processFile(file: File): Promise<{ dataUrl: string; name: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the file."));
    reader.onload = () => {
      const img = new window.Image();
      img.onerror = () => reject(new Error("Could not decode the image."));
      img.onload = () => {
        const maxDim = 1280;
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return reject(new Error("Canvas unavailable."));
        ctx.drawImage(img, 0, 0, w, h);
        resolve({
          dataUrl: canvas.toDataURL("image/jpeg", 0.88),
          name: file.name,
        });
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}

export function UploadZone() {
  const setImage = useGenStore((s) => s.setImage);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const handleFile = useCallback(
    async (file: File | undefined | null) => {
      if (!file) return;
      if (!ACCEPTED.includes(file.type) && !file.name.endsWith(".tif") && !file.name.endsWith(".tiff")) {
        toast.error("Unsupported format. Please use JPG, PNG or WebP.");
        return;
      }
      if (file.size > MAX_BYTES) {
        toast.error("Image is too large. Maximum size is 10 MB.");
        return;
      }
      try {
        const { dataUrl, name } = await processFile(file);
        setImage({
          dataUrl,
          name,
          sizeLabel: formatBytes(Math.round((dataUrl.length * 3) / 4)),
        });
        toast.success(`Loaded ${name}`);
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to load the image."
        );
      }
    },
    [setImage]
  );

  // Paste-from-clipboard support
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? [])[0];
      if (file) {
        e.preventDefault();
        void handleFile(file);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFile]);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a satellite or aerial image"
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current -= 1;
        if (dragDepth.current <= 0) {
          dragDepth.current = 0;
          setDragging(false);
        }
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        void handleFile(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "flex h-full min-h-[190px] sm:min-h-[210px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed text-center px-3 py-3 transition-all outline-none focus-visible:ring-1 focus-visible:ring-amber-400 select-none overflow-hidden",
        dragging
          ? "border-amber-400 bg-amber-400/10"
          : "border-white/15 bg-white/[0.02] hover:border-amber-400/40 hover:bg-white/[0.04]"
      )}
    >
      <CloudUpload
        className={cn(
          "h-7 w-7 shrink-0 transition-colors",
          dragging ? "text-amber-400" : "text-white/60"
        )}
        strokeWidth={1.5}
      />
      <p className="mt-2 text-[13px] font-semibold text-white/90 leading-tight">
        Drop image here
      </p>
      <p className="text-[11px] text-white/55 leading-tight mt-1">or click to browse</p>
      <p className="text-[9.5px] text-white/40 leading-tight mt-1.5 font-mono">
        JPG, PNG, WebP (Max 10MB)
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        className="hidden"
        onChange={(e) => {
          void handleFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}

import { cn } from "@/lib/utils";

export function GeoCaptionLogo({
  className,
  size = 28,
}: {
  className?: string;
  size?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      className={cn("shrink-0", className)}
    >
      <circle
        cx="16"
        cy="16"
        r="12"
        stroke="#e5a84b"
        strokeWidth="1.2"
        strokeDasharray="2 2"
        opacity="0.6"
      />
      <ellipse
        cx="16"
        cy="16"
        rx="13.5"
        ry="6"
        transform="rotate(-25 16 16)"
        stroke="#f3c66f"
        strokeWidth="1.4"
      />
      <ellipse
        cx="16"
        cy="16"
        rx="7"
        ry="13"
        transform="rotate(15 16 16)"
        stroke="#e5a84b"
        strokeWidth="1.2"
        opacity="0.75"
      />
      <circle cx="16" cy="16" r="3" fill="#e5a84b" />
    </svg>
  );
}

export function GeoCaptionWordmark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "font-display text-[21px] font-semibold tracking-tight bg-gradient-to-r from-amber-100 via-amber-200 to-amber-400 bg-clip-text text-transparent",
        className
      )}
    >
      TerraCap
    </span>
  );
}

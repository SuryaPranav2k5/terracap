"use client";

import Image from "next/image";
import {
  House,
  Image as ImageIcon,
  Layers,
  Database,
  BarChart3,
  LayoutGrid,
  Info,
} from "lucide-react";
import { useScrollSpy, scrollToSection } from "@/hooks/use-scroll-spy";
import { cn } from "@/lib/utils";

const SIDEBAR_ITEMS = [
  { id: "home", label: "Home", icon: House },
  { id: "generate", label: "Generate", icon: ImageIcon },
  { id: "architecture", label: "Architecture", icon: Layers },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "gallery", label: "Gallery", icon: LayoutGrid },
  { id: "about", label: "About", icon: Info },
];

const SECTION_IDS = SIDEBAR_ITEMS.map((i) => i.id);

export function Sidebar() {
  const active = useScrollSpy(SECTION_IDS);

  return (
    <aside
      aria-label="Section navigation"
      className="hidden md:flex w-[96px] shrink-0 flex-col justify-between rounded-xl border border-white/10 bg-[#07090e]/90 p-1.5 backdrop-blur-xl shadow-xl min-h-[600px]"
    >
      <nav className="flex flex-col gap-1">
        {SIDEBAR_ITEMS.map((item) => {
          const isActive =
            active === item.id || (active === "" && item.id === "home");
          return (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              aria-current={isActive ? "true" : undefined}
              className={cn(
                "flex h-8 w-full items-center gap-1.5 rounded-lg px-2 text-left text-[10.5px] transition-all outline-none",
                isActive
                  ? "border border-amber-500/50 bg-gradient-to-r from-amber-500/25 to-amber-900/10 font-semibold text-amber-200 shadow-[0_0_12px_rgba(245,158,11,0.2)] [&_svg]:text-amber-400"
                  : "border border-transparent text-white/55 hover:bg-white/[0.05] hover:text-white [&_svg]:text-white/40"
              )}
            >
              <item.icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
              <span className="whitespace-nowrap">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom globe visual with text matching Image 1 */}
      <div className="relative mt-auto overflow-hidden rounded-lg border border-white/10 p-2 bg-black/50">
        <div className="pointer-events-none absolute -bottom-4 -left-4 -right-4 h-24 opacity-30">
          <Image
            src="/images/sidebar-earth-bg.png"
            alt=""
            fill
            sizes="90px"
            className="object-cover object-bottom"
          />
        </div>
        <div className="relative z-10 flex flex-col gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]" />
          <p className="text-[8.5px] leading-tight text-white/70">
            Bridging<br />
            Satellite Imagery<br />
            and Human Language.
          </p>
        </div>
      </div>
    </aside>
  );
}

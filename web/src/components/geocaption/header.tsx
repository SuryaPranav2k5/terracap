"use client";

import Link from "next/link";
import Image from "next/image";
import { useTheme } from "next-themes";
import { Moon, Sun, Github } from "lucide-react";
import { useScrollSpy, scrollToSection } from "@/hooks/use-scroll-spy";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { id: "home", label: "Home" },
  { id: "generate", label: "Generate" },
  { id: "architecture", label: "Architecture" },
  { id: "dataset", label: "Dataset" },
  { id: "results", label: "Results" },
  { id: "about", label: "About" },
];

const NAV_IDS = NAV_ITEMS.map((i) => i.id);

export function Header() {
  const active = useScrollSpy(NAV_IDS);
  const { resolvedTheme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  };

  return (
    <header className="fixed top-2 inset-x-2 sm:inset-x-3 z-50 h-10 max-w-[1440px] mx-auto rounded-xl border border-white/10 bg-[#07090e]/90 backdrop-blur-xl shadow-lg">
      <div className="flex h-full items-center justify-between px-3 sm:px-4">
        {/* Left: Brand Logo & Wordmark */}
        <div className="flex items-center gap-2">
          <button
            className="inline-flex items-center gap-2 outline-none transition-transform hover:opacity-95"
            onClick={() => scrollToSection("home")}
            aria-label="TerraCap home"
          >
            <div className="relative h-6 w-6 shrink-0">
              <Image
                src="/extracted-logo.png"
                alt="TerraCap Logo"
                fill
                priority
                className="object-contain"
              />
            </div>
            <span className="font-display text-[17px] font-bold tracking-tight bg-gradient-to-r from-[#fae7bc] via-[#e5a84b] to-[#b37e22] bg-clip-text text-transparent">
              TerraCap
            </span>
          </button>
        </div>

        {/* Center: Navigation Links */}
        <nav
          aria-label="Primary"
          className="hidden md:flex items-center gap-6 lg:gap-7"
        >
          {NAV_ITEMS.map((item) => {
            const isActive =
              active === item.id || (active === "" && item.id === "home");
            return (
              <button
                key={item.id}
                onClick={() => scrollToSection(item.id)}
                className={cn(
                  "relative py-0.5 text-[12px] font-medium transition-colors outline-none",
                  isActive
                    ? "text-white font-semibold"
                    : "text-white/60 hover:text-white"
                )}
                aria-current={isActive ? "page" : undefined}
              >
                {item.label}
                {isActive && (
                  <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 flex items-center justify-center pointer-events-none">
                    <div className="h-[1px] w-2.5 bg-gradient-to-r from-transparent to-[#e5a84b]" />
                    <div className="h-1.5 w-1.5 rounded-full bg-[#f6d365] shadow-[0_0_8px_#f59e0b]" />
                    <div className="h-[1px] w-2.5 bg-gradient-to-l from-transparent to-[#e5a84b]" />
                  </div>
                )}
              </button>
            );
          })}
        </nav>

        {/* Right: Controls */}
        <div className="flex items-center gap-2">
          {/* Sun / Moon Theme Toggle Capsule */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="flex h-6.5 items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2 text-white/50 transition-colors hover:border-amber-400/40 hover:text-white"
          >
            <Sun className="h-3 w-3 text-white/30" />
            <Moon className="h-3 w-3 text-amber-300 fill-amber-300/20" />
          </button>

          {/* GitHub Pill Button */}
          <Link
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-6.5 items-center gap-1.5 rounded-full border border-white/15 bg-black/50 px-2.5 text-[11px] font-medium text-white/90 transition-all hover:border-amber-400/50 hover:bg-black/70"
          >
            <Github className="h-3 w-3 text-white" />
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

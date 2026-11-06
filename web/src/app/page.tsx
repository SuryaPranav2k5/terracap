"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { LandingHeader } from "@/components/geocaption/landing-header";
import { HeroLanding } from "@/components/geocaption/hero-landing";
import { WorkflowLanding } from "@/components/geocaption/workflow-landing";
import { WorkspaceModal } from "@/components/geocaption/workspace-modal";
import { ArchitectureModal } from "@/components/geocaption/architecture-modal";
import { useGenStore } from "@/lib/store";

export default function Home() {
  const [studioOpen, setStudioOpen] = useState(false);
  const [archOpen, setArchOpen] = useState(false);
  const loadInitial = useGenStore((s) => s.loadInitial);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  return (
    <div className="relative min-h-screen w-full flex flex-col justify-between overflow-x-hidden bg-[#05080f] text-[#f3f3f6] select-none">
      {/* Background Image: exactly Image 2 (landing-bg.jpg) */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
      >
        <Image
          src="/images/landing-bg-2k.jpg"
          alt="Earth orbital background with aurora, satellite imagery, and mountain river landscape"
          fill
          priority
          unoptimized
          sizes="100vw"
          className="object-cover object-center"
        />
        {/* Subtle gradient vignette to ensure text contrast */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/40" />
      </div>

      {/* Main Page Layout Layer (z-10) */}
      <div className="relative z-10 flex min-h-screen flex-col justify-between">
        {/* Top Navbar */}
        <LandingHeader />

        {/* Hero Section */}
        <HeroLanding
          onOpenGenerate={() => setStudioOpen(true)}
          onOpenArchitecture={() => setArchOpen(true)}
        />

        {/* Bottom Workflow: Informational Pipeline Overview */}
        <WorkflowLanding />
      </div>

      {/* Interactive Studio Modal (Real model inference, attention saliency canvas, upload) */}
      <WorkspaceModal open={studioOpen} onOpenChange={setStudioOpen} />

      {/* Dedicated Model Architecture Modal */}
      <ArchitectureModal open={archOpen} onOpenChange={setArchOpen} />
    </div>
  );
}

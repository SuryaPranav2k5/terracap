import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "next-themes";

export const metadata: Metadata = {
  title: "TerraCap — See the Earth in Words",
  description:
    "Generate detailed, human-like descriptions for high-resolution remote sensing images using deep learning and spatial attention.",
  keywords: [
    "TerraCap",
    "remote sensing",
    "image captioning",
    "satellite imagery",
    "deep learning",
    "spatial attention",
  ],
  icons: {
    icon: "/geocaption-icon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#07070b",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;0,800;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased bg-[#07070b] text-[#f3f3f6] overflow-x-hidden selection:bg-amber-500/30 selection:text-amber-200">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
        >
          {children}
          <Toaster position="bottom-right" richColors closeButton />
        </ThemeProvider>
      </body>
    </html>
  );
}

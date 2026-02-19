import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Coding Model Seeks GPU",
  description:
    "Open source coding LLMs ranked by real-world performance, sized to real hardware.",
  openGraph: {
    title: "Coding Model Seeks GPU",
    description:
      "Open source coding LLMs ranked by real-world performance, sized to real hardware.",
    type: "website",
    url: "https://coding-model-seeks-gpu.adriantineo.com",
    images: [
      {
        url: "https://coding-model-seeks-gpu.adriantineo.com/og-image.png",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Coding Model Seeks GPU",
    description:
      "Open source coding LLMs ranked by real-world performance, sized to real hardware.",
    images: ["https://coding-model-seeks-gpu.adriantineo.com/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("theme");if(t==="dark"||(!t&&window.matchMedia("(prefers-color-scheme:dark)").matches)){document.documentElement.classList.add("dark")}}catch(e){}})()`,
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

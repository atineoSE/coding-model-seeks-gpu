"use client";

import { useSyncExternalStore } from "react";

const DESKTOP_QUERY = "(min-width: 640px)";

function getSnapshot(): boolean {
  return window.matchMedia(DESKTOP_QUERY).matches;
}

function getServerSnapshot(): boolean {
  return false; // SSR renders mobile-first
}

function subscribe(callback: () => void): () => void {
  const mql = window.matchMedia(DESKTOP_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

export function useIsDesktop(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

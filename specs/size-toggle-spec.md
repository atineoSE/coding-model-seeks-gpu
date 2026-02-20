# Responsive Price Matrix — Size Toggle Pattern

## Overview

On mobile viewports, the desktop pricing table (items × sizes) is replaced with a **column-toggle** pattern: a segmented control lets users select one size at a time, reducing the matrix to a single-column list. This eliminates horizontal scrolling while preserving scannability.

## Problem

The desktop price matrix has N columns (one per size). On screens below ~640px, this either overflows horizontally or forces cells to shrink to the point of illegibility. Users lose the ability to quickly scan prices.

## Solution

Decompose the matrix into two layers:

1. **Size selector** — a persistent row of toggle buttons (one per size), only one active at a time.
2. **Item list** — a vertical list showing every item with its price for the currently selected size.

The user picks the dimension they care about (size), and the interface flattens the remaining dimension (items) into a simple scannable list.

## Behavior

### Size Selector

- Renders as a horizontal row of equally-sized buttons, one per size.
- Exactly one size is active at all times. Default: the most commonly ordered size, or the middle option if no data is available.
- Tapping a size instantly updates the list below. No loading state needed — all data is already client-side.
- Labels should use human-readable size names (e.g. "12 oz", "Medium") rather than internal codes.

### Item List

- Each row displays: item name (with optional icon/emoji) on the left, price on the right.
- Prices are right-aligned with tabular numerals for easy vertical scanning.
- Alternating row backgrounds (subtle) improve readability on longer lists.
- If an item is unavailable in the selected size, show a disabled state (e.g. greyed-out text with "—" instead of a price).

### Transitions

- Price values should crossfade or have no transition. Avoid layout shifts — the list structure never changes, only the numbers.
- The active state on the toggle should transition smoothly (background color, ~150ms ease).

## Layout Spec

```
┌─────────────────────────────────┐
│  [ 8oz ][ 12oz ][ 16oz ][ 20oz ]│  ← size selector, full width
├─────────────────────────────────┤
│  ☕ Espresso              $4.00 │  ← item row
│  🥛 Cappuccino            $4.60 │
│  🧊 Cold Brew             $5.00 │
│  🍵 Matcha Latte          $5.60 │
│  🍫 Mocha                 $5.40 │
│  ...                            │
└─────────────────────────────────┘
```

### Sizing

| Element | Value |
|---|---|
| Selector height | 40px |
| Selector button font | 12px, bold, uppercase tracking |
| Gap between selector and list | 12px |
| Row height | 48–52px |
| Row padding | 14px horizontal |
| Item name font | 14px, semibold |
| Price font | 16–17px, bold, tabular-nums |

## Edge Cases

- **1–2 sizes only**: Still use the toggle for consistency, but buttons will be wider. If there is only one size, skip the selector entirely and show the plain list.
- **Many sizes (6+)**: Allow the selector to scroll horizontally, or group sizes into categories (e.g. "Small / Medium / Large" with a secondary detail label).
- **Long item names**: Truncate with ellipsis. The price column has a fixed minimum width; the name column takes the remaining space.
- **Empty state**: If no items exist for a category, show a centered message within the list area.
- **Price changes / promotions**: A small strikethrough original price can appear above or beside the current price, but keep it compact — no more than two lines per row.

## Accessibility

- The size selector should use `role="tablist"` with each button as `role="tab"` and `aria-selected` on the active one.
- The item list should use `role="tabpanel"` linked to the active tab via `aria-labelledby`.
- Arrow keys should navigate between sizes when the selector is focused.
- Price values should be announced with their currency (e.g. `aria-label="Espresso, four dollars"`).

## When to Use This Pattern

**Good fit:**
- Users typically know what size they want before browsing items.
- The primary comparison axis is across items (which drink?), not across sizes.
- There are more items than sizes.

**Poor fit:**
- Users are deciding between sizes for a single item (use accordion or detail-expand instead).
- Sizes have complex metadata beyond just a label (e.g. nutritional info per size) — a card-based layout may be better.

## Breakpoint Strategy

| Viewport | Behavior |
|---|---|
| ≥ 640px | Desktop matrix table (no change) |
| < 640px | Size Toggle pattern |

The switch should be handled via CSS media query or a responsive hook. Both layouts read from the same data source; only the presentation changes.

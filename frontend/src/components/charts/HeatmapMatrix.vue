<template>
  <section class="heatmap-shell">
    <header v-if="title || subtitle" class="heatmap-head">
      <div>
        <h3 v-if="title">{{ title }}</h3>
        <p v-if="subtitle">{{ subtitle }}</p>
      </div>
      <span v-if="badge" class="heatmap-badge">{{ badge }}</span>
    </header>

    <div v-if="rows.length && cols.length" class="heatmap-grid">
      <div class="heatmap-corner"></div>
      <div v-for="col in cols" :key="col" class="heatmap-col-label">{{ col }}</div>
      <template v-for="(row, rIndex) in rows" :key="row">
        <div class="heatmap-row-label">{{ row }}</div>
        <button
          v-for="(col, cIndex) in cols"
          :key="`${row}-${col}`"
          class="heatmap-cell"
          :style="cellStyle(valueAt(rIndex, cIndex))"
          type="button"
          :title="`${row} vs ${col}: ${formatValue(valueAt(rIndex, cIndex))}`"
        >
          {{ formatValue(valueAt(rIndex, cIndex)) }}
        </button>
      </template>
    </div>

    <div v-if="rows.length && cols.length" class="heatmap-legend" aria-label="Agreement intensity legend">
      <span>Lower agreement</span>
      <span class="legend-scale" aria-hidden="true"></span>
      <span>Higher agreement</span>
    </div>

    <div v-else class="heatmap-empty">
      <strong>{{ emptyTitle }}</strong>
      <p>{{ emptyText }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  badge: { type: String, default: '' },
  rows: { type: Array, default: () => [] },
  cols: { type: Array, default: () => [] },
  values: { type: Array, default: () => [] },
  emptyTitle: { type: String, default: 'No matrix data yet' },
  emptyText: { type: String, default: 'Run a comparison to populate the agreement matrix.' },
})

function valueAt(rowIndex, colIndex) {
  return props.values[rowIndex]?.[colIndex]
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') return `${Math.round(value * 100)}%`
  return String(value)
}

function cellStyle(value) {
  const normalized = typeof value === 'number' ? Math.max(0, Math.min(1, value)) : Number.parseFloat(String(value).replace('%', '')) / 100
  const safe = Number.isFinite(normalized) ? normalized : 0
  const lightness = 97 - safe * 36
  return {
    backgroundColor: `hsl(211 48% ${lightness}%)`,
    color: safe > 0.62 ? '#FFFFFF' : '#1E3A5F',
    borderColor: `hsla(211 48% 34% / ${0.14 + safe * 0.34})`,
  }
}
</script>

<style scoped>
.heatmap-shell {
  display: grid;
  gap: 12px;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.94));
  box-shadow: var(--shadow-soft);
}

.heatmap-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.heatmap-head h3 {
  margin: 0;
  color: var(--c-text-primary);
  font-size: 16px;
  font-weight: 700;
}

.heatmap-head p {
  margin: 4px 0 0;
  color: var(--c-text-secondary);
  font-size: 13px;
}

.heatmap-badge {
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 12px;
  font-weight: 700;
}

.heatmap-grid {
  display: grid;
  grid-template-columns: 110px repeat(auto-fit, minmax(52px, 1fr));
  gap: 8px;
  align-items: stretch;
}

.heatmap-corner,
.heatmap-col-label,
.heatmap-row-label,
.heatmap-cell {
  min-height: 42px;
  display: grid;
  place-items: center;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.heatmap-col-label,
.heatmap-row-label {
  padding: 8px;
  background: rgba(255, 255, 255, 0.78);
  color: var(--c-text-secondary);
  font-size: 12px;
  font-weight: 700;
  text-align: center;
}

.heatmap-row-label {
  justify-content: start;
  text-align: left;
}

.heatmap-cell {
  padding: 8px 6px;
  font-size: 12px;
  font-weight: 700;
  border-width: 1px;
}

.heatmap-empty {
  min-height: 180px;
  display: grid;
  place-items: center;
  padding: 24px;
  border: 1px dashed var(--c-border-strong);
  border-radius: 16px;
  color: var(--c-text-secondary);
  text-align: center;
  background: rgba(255, 255, 255, 0.66);
}

.heatmap-empty strong {
  color: var(--c-text-primary);
}

.heatmap-empty p {
  margin: 6px 0 0;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  color: var(--c-text-secondary);
  font-size: .72rem;
}
.legend-scale {
  width: 112px;
  height: 8px;
  border: 1px solid rgba(49, 92, 140, .18);
  border-radius: 999px;
  background: linear-gradient(90deg, hsl(211 48% 97%), hsl(211 48% 61%));
}

</style>

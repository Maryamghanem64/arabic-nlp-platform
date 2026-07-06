<template>
  <section class="chart-shell" :class="{ empty: !hasData }">
    <header v-if="title || subtitle" class="chart-head">
      <div>
        <h3 v-if="title">{{ title }}</h3>
        <p v-if="subtitle">{{ subtitle }}</p>
      </div>
      <span v-if="badge" class="chart-badge">{{ badge }}</span>
    </header>

    <div v-if="hasData" class="chart-canvas-wrap" :style="{ height: `${height}px` }">
      <canvas ref="canvasEl" :aria-label="ariaLabel" role="img"></canvas>
    </div>

    <div v-else class="chart-empty">
      <strong>{{ emptyTitle }}</strong>
      <p>{{ emptyText }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import Chart from 'chart.js/auto'
import { CONFIDENCE_COLORS } from '@/constants/designTokens'

const props = defineProps({
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  badge: { type: String, default: '' },
  type: { type: String, default: 'bar' },
  labels: { type: Array, default: () => [] },
  datasets: { type: Array, default: () => [] },
  height: { type: Number, default: 280 },
  stacked: { type: Boolean, default: false },
  indexAxis: { type: String, default: 'x' },
  ariaLabel: { type: String, default: 'Scientific chart' },
  emptyTitle: { type: String, default: 'No chart data yet' },
  emptyText: { type: String, default: 'Run an analysis to populate this visualization.' },
})

const canvasEl = ref(null)
let chartInstance = null

const hasData = computed(() => props.labels.length > 0 && props.datasets.some((dataset) => Array.isArray(dataset.data) && dataset.data.length))

function palette(index) {
  // Restrained research palette. Semantic status colors should be supplied
  // explicitly by the calling view rather than inferred from dataset order.
  const colors = [
    '#315C8C',
    '#5B6F8A',
    '#5F7F78',
    '#7A6F8F',
    '#8A7357',
    '#64748B',
  ]
  return colors[index % colors.length]
}

function buildChart() {
  if (!canvasEl.value || !hasData.value) return

  chartInstance?.destroy()
  const context = canvasEl.value.getContext('2d')
  if (!context) return

  chartInstance = new Chart(context, {
    type: props.type,
    data: {
      labels: props.labels,
      datasets: props.datasets.map((dataset, index) => {
        const color = dataset.borderColor || palette(index)
        const fillColor = dataset.backgroundColor || `${color}22`
        return {
          borderWidth: 2,
          tension: 0.22,
          pointRadius: 2,
          pointHoverRadius: 4,
          ...dataset,
          borderColor: color,
          backgroundColor: fillColor,
          fill: dataset.fill ?? props.type === 'line',
        }
      }),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: props.indexAxis,
      animation: {
        duration: 700,
      },
      scales:
        props.type === 'radar'
          ? undefined
          : {
              x: {
                stacked: props.stacked,
                ticks: { color: '#64748B' },
                grid: { color: 'rgba(148, 163, 184, 0.18)' },
              },
              y: {
                stacked: props.stacked,
                beginAtZero: true,
                ticks: { color: '#64748B' },
                grid: { color: 'rgba(148, 163, 184, 0.18)' },
              },
            },
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            color: '#475569',
            usePointStyle: true,
            boxWidth: 8,
            boxHeight: 8,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: '#0F172A',
          titleColor: '#FFFFFF',
          bodyColor: '#E2E8F0',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
        },
      },
    },
  })
}

onMounted(buildChart)
watch(
  () => [props.type, props.labels, props.datasets, props.stacked, props.indexAxis],
  () => buildChart(),
  { deep: true },
)

onBeforeUnmount(() => {
  chartInstance?.destroy()
})
</script>

<style scoped>
.chart-shell {
  display: grid;
  gap: 12px;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.94));
  box-shadow: var(--shadow-soft);
}

.chart-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.chart-head h3 {
  margin: 0;
  color: var(--c-text-primary);
  font-size: 16px;
  font-weight: 700;
}

.chart-head p {
  margin: 4px 0 0;
  color: var(--c-text-secondary);
  font-size: 13px;
}

.chart-badge {
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 12px;
  font-weight: 700;
}

.chart-canvas-wrap {
  position: relative;
  width: 100%;
}

.chart-empty {
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

.chart-empty strong {
  color: var(--c-text-primary);
}

.chart-empty p {
  margin: 6px 0 0;
}
</style>

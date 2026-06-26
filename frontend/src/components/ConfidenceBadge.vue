<template>
  <span class="conf-badge" :style="badgeStyle">
    <span class="conf-dot" :style="{ backgroundColor: color.dot }"></span>
    <span>{{ displayText }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue'
import { CONFIDENCE_COLORS, getConfidenceLevel } from '@/constants/designTokens'

const props = defineProps({
  level: {
    type: String,
    default: null,
  },
  score: {
    type: Number,
    default: null,
  },
})

const resolvedLevel = computed(() => {
  if (props.level) return props.level
  if (props.score !== null) return getConfidenceLevel(props.score)
  return 'low'
})

const color = computed(() => CONFIDENCE_COLORS[resolvedLevel.value] || CONFIDENCE_COLORS.low)

const badgeStyle = computed(() => ({
  backgroundColor: color.value.bg,
  borderColor: color.value.border,
  color: color.value.text,
}))

const displayText = computed(() => {
  if (props.score !== null) return `${Math.round(props.score * 100)}%`
  return color.value.label
})
</script>

<style scoped>
.conf-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 10px;
  border: 1px solid;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  line-height: 1.6;
}

.conf-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
</style>

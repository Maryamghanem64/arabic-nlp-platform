<template>
  <span
    class="tool-badge"
    :style="{
      backgroundColor: colors.bg,
      borderColor: colors.border,
      color: colors.text,
    }"
  >
    {{ colors.label }}
  </span>
</template>

<script setup>
import { computed } from 'vue'
import { TOOL_COLORS } from '@/constants/designTokens'

const props = defineProps({
  tool: {
    type: String,
    default: 'camel',
  },
  showGroup: {
    type: Boolean,
    default: false,
  },
})

const colors = computed(() => TOOL_COLORS[props.tool] || TOOL_COLORS.camel)
</script>

<style scoped>
.tool-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  border: 1px solid;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  line-height: 1.6;
}
</style>

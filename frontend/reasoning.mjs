const EFFORT_MAP = {
  kimi: { low: 'low', medium: 'high', high: 'max', max: 'max' },
  qwen: { low: 'low', medium: 'medium', high: 'xhigh', max: 'xhigh' },
  thinking: { low: 'low', medium: 'high', high: 'max', max: 'max' },
  openai: { low: 'low', medium: 'medium', high: 'high', max: 'max' },
};

const OPTION_MAP = {
  kimi: [
    ['low', 'low', '低 low'], ['medium', 'high', '高 high'], ['max', 'max', '最高 max'],
  ],
  qwen: [
    ['low', 'low', '低 low'], ['medium', 'medium', '中 medium'], ['high', 'xhigh', '高 xhigh'],
  ],
  thinking: [
    ['low', 'low', '低 low'], ['medium', 'high', '高 high'], ['max', 'max', '最高 max'],
  ],
  openai: [
    ['low', 'low', '低 low'], ['medium', 'medium', '中 medium'],
    ['high', 'high', '高 high'], ['max', 'max', '最高 max'],
  ],
};

export function reasoningOptions(profile) {
  return (OPTION_MAP[profile?.request_dialect] || []).map(([value, native, label]) => ({ value, native, label }));
}

export function effectiveReasoning(profile, selection = 'profile') {
  if (!profile) return null;
  const map = EFFORT_MAP[profile.request_dialect];
  if (map && selection && selection !== 'profile' && map[selection]) return map[selection];
  return profile.reasoning_effort || null;
}

export function reasoningSelection(profile, selection) {
  if (!selection || selection === 'profile') return 'profile';
  const options = reasoningOptions(profile);
  if (options.some(option => option.value === selection)) return selection;
  const native = effectiveReasoning(profile, selection);
  return options.find(option => option.native === native)?.value || 'profile';
}

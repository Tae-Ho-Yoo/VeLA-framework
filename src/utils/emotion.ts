import type { EmotionLabel } from './types'

export function computeEmotionLabel(valence: number, arousal: number): EmotionLabel {
  const vHigh = valence >= 5
  const aHigh = arousal >= 5

  if (vHigh && aHigh) return 'HVHA'
  if (vHigh && !aHigh) return 'HVLA'
  if (!vHigh && aHigh) return 'LVHA'
  return 'LVLA'
}


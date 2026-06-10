import { create } from 'zustand'
import type { ParticipantInfo, TLXRatings, SurveyRecord } from '../utils/types'

type SurveyState = {
  participant: ParticipantInfo | null
  nasaTlx: TLXRatings
  committedRecord: SurveyRecord | null
  setParticipant: (p: ParticipantInfo) => void
  setNasaTlx: (patch: Partial<TLXRatings>) => void
  commitRecord: () => SurveyRecord | null
  logout: () => void
  startNewSurveyRound: () => void
  buildRecord: () => SurveyRecord | null
}

const defaultNasaTlx: TLXRatings = {
  mental_demand: 5,
  physical_demand: 5,
  temporal_demand: 5,
  performance: 5,
  effort: 5,
  frustration: 5,
}

function meanTLX(tlx: TLXRatings) {
  const values = Object.values(tlx)
  return values.reduce((a, b) => a + b, 0) / values.length
}

function makeRecord(participant: ParticipantInfo, nasaTlx: TLXRatings): SurveyRecord {
  return {
    created_at: new Date().toISOString(),
    participant,
    nasa_tlx: nasaTlx,
    nasa_tlx_mean: meanTLX(nasaTlx),
    metadata: {
      app: 'vela-survey-web',
      schema_version: 3,
      scale: 'NASA-TLX',
    },
  }
}

export const useSurveyStore = create<SurveyState>((set, get) => ({
  participant: null,
  nasaTlx: defaultNasaTlx,
  committedRecord: null,

  setParticipant: (p) => set({ participant: p }),

  setNasaTlx: (patch) =>
    set({ nasaTlx: { ...get().nasaTlx, ...patch } }),

  commitRecord: () => {
    const { participant, nasaTlx } = get()
    if (!participant) return null
    const record = makeRecord(participant, nasaTlx)
    set({ committedRecord: record })
    return record
  },

  logout: () =>
    set({
      participant: null,
      nasaTlx: defaultNasaTlx,
      committedRecord: null,
    }),

  startNewSurveyRound: () =>
    set({
      nasaTlx: { ...defaultNasaTlx },
      committedRecord: null,
    }),

  buildRecord: () => {
    const { participant, nasaTlx } = get()
    if (!participant) return null
    return makeRecord(participant, nasaTlx)
  },
}))
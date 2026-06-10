export type Gender = 'male' | 'female' | 'other' | 'prefer_not_to_say'

export type TLXDimension =
  | 'mental_demand'
  | 'physical_demand'
  | 'temporal_demand'
  | 'performance'
  | 'effort'
  | 'frustration'

export type ParticipantInfo = {
  participant_id: string
  name: string
  age: number
  gender: Gender
}

export type TLXRatings = Record<TLXDimension, number> // 1..9

export type SurveyRecord = {
  created_at: string
  participant: ParticipantInfo
  nasa_tlx: TLXRatings
  nasa_tlx_mean: number
  metadata: {
    app: 'vela-survey-web'
    schema_version: 3
    scale: 'NASA-TLX'
  }
}
import type { Gender } from './types'

export type IntroFormValues = {
  name: string
  age: string
  gender: Gender | ''
  participantId: string
}

export type IntroFormErrors = Partial<Record<keyof IntroFormValues, string>>

export function validateIntro(values: IntroFormValues): IntroFormErrors {
  const errors: IntroFormErrors = {}

  const name = values.name.trim()
  if (!name) errors.name = '이름을 입력해주세요.'

  const participantId = values.participantId.trim()
  if (!participantId) errors.participantId = '참가 번호를 입력해주세요.'
  if (participantId.length > 64) errors.participantId = '번호가 너무 깁니다.'

  const age = Number(values.age)
  if (!values.age.trim()) errors.age = '나이를 입력해주세요.'
  else if (!Number.isFinite(age) || !Number.isInteger(age)) errors.age = '숫자만 입력해주세요. (예: 25)'
  else if (age < 1 || age > 120) errors.age = '나이는 1~120 사이로 입력해주세요.'

  if (!values.gender) errors.gender = '성별을 선택해주세요.'

  return errors
}

export function isEmptyErrors(errors: Record<string, string | undefined>): boolean {
  return Object.values(errors).every((v) => !v)
}


import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card } from '../components/Card'
import { Field } from '../components/Field'
import { Input } from '../components/Input'
import { Select } from '../components/Select'
import { Button } from '../components/Button'
import { useSurveyStore } from '../store/surveyStore'
import type { Gender } from '../utils/types'
import { isEmptyErrors, validateIntro, type IntroFormValues } from '../utils/validation'

const genderOptions: Array<{ value: Gender; label: string }> = [
  { value: 'male', label: '남성' },
  { value: 'female', label: '여성' },
  { value: 'other', label: '기타' },
  { value: 'prefer_not_to_say', label: '응답하지 않음' },
]

export function IntroPage() {
  const navigate = useNavigate()
  const setParticipant = useSurveyStore((s) => s.setParticipant)
  const participant = useSurveyStore((s) => s.participant)

  useEffect(() => {
    if (participant) navigate('/survey', { replace: true })
  }, [participant, navigate])

  const [values, setValues] = useState<IntroFormValues>({
    name: '',
    age: '',
    gender: '',
    participantId: '',
  })

  const errors = useMemo(() => validateIntro(values), [values])

  const canSubmit = isEmptyErrors(errors)

  if (participant) return null

  return (
    <div className="container-page space-y-6">
      <div className="space-y-2">
        <div className="text-2xl font-semibold text-zinc-900">시작하기</div>
        <div className="text-sm text-zinc-600">
          아래 항목을 적어 주시면 설문으로 이어집니다. 빈칸 없이 모두 적어 주세요.
        </div>
      </div>

      <Card title="기본 정보" subtitle="아래 네 가지를 모두 입력해 주세요.">
        <div className="space-y-4">
          <Field label="이름" error={errors.name}>
            <Input
              value={values.name}
              hasError={Boolean(errors.name)}
              onChange={(e) => setValues((v) => ({ ...v, name: e.target.value }))}
              placeholder="홍길동"
              autoComplete="off"
            />
          </Field>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Field label="나이" error={errors.age} hint="숫자만, 예: 25">
              <Input
                value={values.age}
                hasError={Boolean(errors.age)}
                onChange={(e) => setValues((v) => ({ ...v, age: e.target.value }))}
                placeholder="25"
                inputMode="numeric"
                autoComplete="off"
              />
            </Field>

            <Field label="성별" error={errors.gender}>
              <Select
                value={values.gender}
                hasError={Boolean(errors.gender)}
                onChange={(e) => setValues((v) => ({ ...v, gender: e.target.value as Gender }))}
              >
                <option value="">선택</option>
                {genderOptions.map((g) => (
                  <option key={g.value} value={g.value}>
                    {g.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>

          <Field label="참가 번호" error={errors.participantId} hint="실험에서 안내한 번호 (예: 1번, S01)">
            <Input
              value={values.participantId}
              hasError={Boolean(errors.participantId)}
              onChange={(e) => setValues((v) => ({ ...v, participantId: e.target.value }))}
              placeholder="S001"
              autoComplete="off"
            />
          </Field>

          <div className="flex items-center justify-end gap-2 pt-2">
            <Button
              type="button"
              onClick={() => {
                if (!canSubmit) return
                setParticipant({
                  participant_id: values.participantId.trim(),
                  name: values.name.trim(),
                  age: Number(values.age),
                  gender: values.gender as Gender,
                })
                navigate('/survey')
              }}
              disabled={!canSubmit}
            >
              설문으로 가기
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}


import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card } from '../components/Card'
import { Button } from '../components/Button'
import { TLXRadio } from '../components/TLXRadio'
import { useSurveyStore } from '../store/surveyStore'

export function SurveyPage() {
  const navigate = useNavigate()

  const participant = useSurveyStore((s) => s.participant)
  const nasaTlx = useSurveyStore((s) => s.nasaTlx)
  const setNasaTlx = useSurveyStore((s) => s.setNasaTlx)
  const commitRecord = useSurveyStore((s) => s.commitRecord)

  useEffect(() => {
    if (!participant) {
      navigate('/', { replace: true })
    }
  }, [participant, navigate])

  if (!participant) return null

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <Card>
        <h1 className="text-3xl font-bold">NASA-TLX 설문</h1>

        <p className="mt-3 text-zinc-600">
          방금 수행한 작업에 대해 각 문항을 1점~9점으로 평가해 주세요.
        </p>

        <div className="mt-8 space-y-10">
          <TLXRadio
            label="정신적 요구도"
            description="생각, 판단, 기억, 집중이 얼마나 필요했나요?"
            value={nasaTlx.mental_demand}
            onChange={(v) => setNasaTlx({ mental_demand: v })}
            leftLabel="매우 낮음"
            rightLabel="매우 높음"
          />

          <TLXRadio
            label="신체적 요구도"
            description="움직임이나 신체적 노력이 얼마나 필요했나요?"
            value={nasaTlx.physical_demand}
            onChange={(v) => setNasaTlx({ physical_demand: v })}
            leftLabel="매우 낮음"
            rightLabel="매우 높음"
          />

          <TLXRadio
            label="시간적 요구도"
            description="시간 압박이나 속도 압박을 얼마나 느꼈나요?"
            value={nasaTlx.temporal_demand}
            onChange={(v) => setNasaTlx({ temporal_demand: v })}
            leftLabel="매우 낮음"
            rightLabel="매우 높음"
          />

          <TLXRadio
            label="수행도"
            description="작업을 얼마나 성공적으로 수행했다고 느꼈나요?"
            value={nasaTlx.performance}
            onChange={(v) => setNasaTlx({ performance: v })}
            leftLabel="매우 성공적"
            rightLabel="매우 실패함"
          />

          <TLXRadio
            label="노력"
            description="원하는 수행 수준을 달성하기 위해 얼마나 노력했나요?"
            value={nasaTlx.effort}
            onChange={(v) => setNasaTlx({ effort: v })}
            leftLabel="매우 낮음"
            rightLabel="매우 높음"
          />

          <TLXRadio
            label="좌절감"
            description="짜증, 불안, 스트레스, 좌절감을 얼마나 느꼈나요?"
            value={nasaTlx.frustration}
            onChange={(v) => setNasaTlx({ frustration: v })}
            leftLabel="매우 낮음"
            rightLabel="매우 높음"
          />
        </div>

        <div className="mt-10 flex justify-end">
          <Button
            onClick={() => {
              commitRecord()
              navigate('/result')
            }}
          >
            결과 보기
          </Button>
        </div>
      </Card>
    </div>
  )
}
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileJson } from 'lucide-react'
import { Card } from '../components/Card'
import { Button } from '../components/Button'
import { useSurveyStore } from '../store/surveyStore'
import { submitRecord } from '../utils/api'

export function ResultPage() {
  const navigate = useNavigate()
  const startNewSurveyRound = useSurveyStore((s) => s.startNewSurveyRound)
  const record = useSurveyStore((s) => s.committedRecord)
  const participant = useSurveyStore((s) => s.participant)

  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState<string | null>(null)
  const submitOnceRef = useRef(false)

  useEffect(() => {
    if (!record || submitOnceRef.current) return
    submitOnceRef.current = true

    setSaveState('saving')
    setSaveError(null)
    submitRecord(record).then((res) => {
      if (res.ok) {
        setSaveState('saved')
      } else {
        setSaveState('error')
        setSaveError(res.error)
        submitOnceRef.current = false
      }
    })
  }, [record])

  if (!participant) {
    return (
      <div className="container-page">
        <Card title="결과" subtitle="먼저 첫 화면에서 정보를 입력해 주세요.">
          <div className="flex justify-end">
            <Button type="button" onClick={() => navigate('/')}>
              첫 화면으로
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  if (!record) {
    return (
      <div className="container-page">
        <Card title="결과" subtitle="설문을 마친 뒤 설문 화면에서 「결과 보기」를 눌러 주세요.">
          <div className="flex justify-end">
            <Button type="button" onClick={() => navigate('/survey')}>
              설문으로
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="container-page space-y-6">
      <div className="space-y-2">
        <div className="text-2xl font-semibold text-zinc-900">저장 완료</div>
        <div className="space-y-1 text-sm text-zinc-600">
          <p>
            아래 데이터는 <span className="font-medium text-zinc-800">서버에 자동으로 저장</span>됩니다.
          </p>
          <p>
            같은 참가자로 반복하려면 「다시 설문하기」, 다른 사람이면 상단 「로그아웃」을 사용하세요.
          </p>
        </div>
        {saveState === 'saving' ? (
          <p className="text-sm text-zinc-500">서버에 저장하는 중…</p>
        ) : null}
        {saveState === 'saved' ? (
          <p className="text-sm text-emerald-700">서버에 저장되었습니다.</p>
        ) : null}
        {saveState === 'error' ? (
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <p className="text-sm text-red-600">
              서버 저장에 실패했습니다. 터미널에서 <code className="rounded bg-zinc-100 px-1">npm run server</code> 또는{' '}
              <code className="rounded bg-zinc-100 px-1">npm run dev:full</code> 인지 확인해 주세요. ({saveError})
            </p>
            <Button
              type="button"
              variant="secondary"
              className="shrink-0"
              onClick={() => {
                setSaveState('saving')
                setSaveError(null)
                submitRecord(record).then((res) => {
                  if (res.ok) setSaveState('saved')
                  else {
                    setSaveState('error')
                    setSaveError(res.error)
                  }
                })
              }}
            >
              다시 시도
            </Button>
          </div>
        ) : null}
      </div>

      <Card
        title="저장된 결과"
        subtitle={`감정 구분 코드: ${record.emotion_label}`}
        right={
          <div className="inline-flex items-center gap-2 text-xs text-zinc-500">
            <FileJson className="h-4 w-4" /> 형식 v{record.metadata.schema_version}
          </div>
        }
      >
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              onClick={() => {
                startNewSurveyRound()
                navigate('/survey')
              }}
            >
              다시 설문하기
            </Button>
          </div>

          <pre className="max-h-[60vh] overflow-auto rounded-lg border border-zinc-200 bg-zinc-950 p-4 text-xs leading-relaxed text-zinc-100">
            {JSON.stringify(record, null, 2)}
          </pre>
        </div>
      </Card>
    </div>
  )
}

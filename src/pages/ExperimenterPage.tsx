import { useEffect, useState } from 'react'
import { Card } from '../components/Card'
import { Button } from '../components/Button'
import { Input } from '../components/Input'

type Criterion = {
  code: 'text_color' | 'word_meaning'
  label: string
  speech: string
}

type StroopLog = {
  trial: number
  participantId: string
  sessionId: string
  criterion_code: string
  criterion_label: string
  spoken_instruction: string
  timestamp: string
}

const CRITERIA: Criterion[] = [
  {
    code: 'text_color',
    label: '글자색 기준',
    speech: '글자색으로 분류하세요',
  },
  {
    code: 'word_meaning',
    label: '의미색 기준',
    speech: '의미색으로 분류하세요',
  },
]

const LOCAL_STORAGE_KEY = 'stroop_experimenter_logs'

function loadLocalLogs(): StroopLog[] {
  const saved = localStorage.getItem(LOCAL_STORAGE_KEY)

  if (!saved) return []

  try {
    return JSON.parse(saved)
  } catch {
    return []
  }
}

function saveLocalLogs(logs: StroopLog[]) {
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(logs))
}

function playBeep() {
  const AudioContextClass =
    window.AudioContext || (window as any).webkitAudioContext

  const audioCtx = new AudioContextClass()
  const oscillator = audioCtx.createOscillator()
  const gainNode = audioCtx.createGain()

  oscillator.type = 'sine'
  oscillator.frequency.setValueAtTime(800, audioCtx.currentTime)

  gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime)

  oscillator.connect(gainNode)
  gainNode.connect(audioCtx.destination)

  oscillator.start()
  oscillator.stop(audioCtx.currentTime + 0.25)
}

function speakKorean(text: string) {
  window.speechSynthesis.cancel()

  const utterance = new SpeechSynthesisUtterance(text)

  utterance.lang = 'ko-KR'
  utterance.rate = 0.9
  utterance.pitch = 1.0
  utterance.volume = 1.0

  window.speechSynthesis.speak(utterance)
}

function convertLogsToCSV(logs: StroopLog[]) {
  const header = [
    'trial',
    'participantId',
    'sessionId',
    'criterion_code',
    'criterion_label',
    'spoken_instruction',
    'timestamp',
  ]

  const rows = logs.map((log) => [
    log.trial,
    log.participantId,
    log.sessionId,
    log.criterion_code,
    log.criterion_label,
    log.spoken_instruction,
    log.timestamp,
  ])

  return [
    header.join(','),
    ...rows.map((row) =>
      row
        .map((value) => {
          const escaped = String(value).replaceAll('"', '""')
          return `"${escaped}"`
        })
        .join(',')
    ),
  ].join('\n')
}

function makeFilename(baseName: string, extension: string) {
  const now = new Date()

  const yyyy = now.getFullYear()
  const mm = String(now.getMonth() + 1).padStart(2, '0')
  const dd = String(now.getDate()).padStart(2, '0')
  const hh = String(now.getHours()).padStart(2, '0')
  const min = String(now.getMinutes()).padStart(2, '0')
  const ss = String(now.getSeconds()).padStart(2, '0')

  return `${baseName}_${yyyy}${mm}${dd}_${hh}${min}${ss}.${extension}`
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')

  a.href = url
  a.download = filename

  document.body.appendChild(a)
  a.click()

  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function ExperimenterPage() {
  const [logs, setLogs] = useState<StroopLog[]>([])
  const [selectedCriterion, setSelectedCriterion] = useState(
    '아직 분류 기준이 지정되지 않았습니다'
  )
  const [participantId, setParticipantId] = useState('P001')
  const [sessionId, setSessionId] = useState('S01')
  const [serverStatus, setServerStatus] = useState('대기 중')

  useEffect(() => {
    const savedLogs = loadLocalLogs()
    setLogs(savedLogs)

    if (savedLogs.length > 0) {
      setSelectedCriterion(savedLogs[savedLogs.length - 1].criterion_label)
    }
  }, [])

  function handleBeepOnly() {
    playBeep()
  }

  async function saveLogToServer(log: StroopLog) {
    try {
      const response = await fetch('/api/stroop-log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(log),
      })

      if (!response.ok) {
        throw new Error('서버 저장 실패')
      }

      setServerStatus('서버 저장 완료')
    } catch (error) {
      console.error(error)
      setServerStatus('서버 저장 실패: 로컬에는 저장됨')
    }
  }

  async function handleRandomCriterion() {
    window.speechSynthesis.cancel()

    const randomIndex = Math.floor(Math.random() * CRITERIA.length)
    const selected = CRITERIA[randomIndex]

    const newLog: StroopLog = {
      trial: logs.length + 1,
      participantId,
      sessionId,
      criterion_code: selected.code,
      criterion_label: selected.label,
      spoken_instruction: selected.speech,
      timestamp: new Date().toISOString(),
    }

    const updatedLogs = [...logs, newLog]

    setLogs(updatedLogs)
    saveLocalLogs(updatedLogs)

    setSelectedCriterion(selected.label)
    setServerStatus('저장 중...')

    await saveLogToServer(newLog)

    speakKorean(selected.speech)
  }

  function downloadCSV() {
    if (logs.length === 0) {
      alert('저장된 로그가 없습니다.')
      return
    }

    const csvContent = convertLogsToCSV(logs)

    const blob = new Blob(['\uFEFF' + csvContent], {
      type: 'text/csv;charset=utf-8;',
    })

    downloadBlob(blob, makeFilename('stroop_instruction_logs', 'csv'))
  }

  function downloadJSON() {
    if (logs.length === 0) {
      alert('저장된 로그가 없습니다.')
      return
    }

    const jsonContent = JSON.stringify(logs, null, 2)

    const blob = new Blob([jsonContent], {
      type: 'application/json;charset=utf-8;',
    })

    downloadBlob(blob, makeFilename('stroop_instruction_logs', 'json'))
  }

  function resetLocalLogs() {
    const ok = confirm('정말 실험자용 로컬 로그를 삭제할까요?')

    if (!ok) return

    setLogs([])
    saveLocalLogs([])
    setSelectedCriterion('아직 분류 기준이 지정되지 않았습니다')
    setServerStatus('로컬 로그 초기화 완료')
  }

  return (
    <div className="container-page space-y-6">
      <div className="space-y-2">
        <div className="text-2xl font-semibold text-zinc-900">
          실험자용 Stroop 분류 기준 안내
        </div>

        <div className="text-sm text-zinc-600">
          Beep 버튼과 분류 기준 랜덤 안내 버튼을 따로 사용합니다.
        </div>
      </div>

      <Card title="실험 정보" subtitle="현재 피험자와 세션 번호를 입력하세요.">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-sm font-medium text-zinc-700">피험자 ID</div>
            <Input
              value={participantId}
              onChange={(e) => setParticipantId(e.target.value)}
              placeholder="예: P001"
              autoComplete="off"
            />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium text-zinc-700">Session ID</div>
            <Input
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="예: S01"
              autoComplete="off"
            />
          </div>
        </div>
      </Card>

      <Card>
        <div className="flex flex-col items-center gap-6">
          <div className="grid w-full grid-cols-1 gap-4 md:grid-cols-2">
            <Button
              type="button"
              onClick={handleBeepOnly}
              className="px-10 py-8 text-2xl"
            >
              Beep 소리
            </Button>

            <Button
              type="button"
              onClick={handleRandomCriterion}
              className="px-10 py-8 text-2xl"
            >
              분류 기준 랜덤 안내
            </Button>
          </div>

          <div className="text-4xl font-bold text-zinc-900">
            {selectedCriterion}
          </div>

          <div className="text-lg text-zinc-600">현재 Trial: {logs.length}</div>

          <div className="text-sm text-zinc-500">저장 상태: {serverStatus}</div>

          <div className="flex flex-wrap justify-center gap-2">
            <Button type="button" variant="secondary" onClick={downloadCSV}>
              CSV 다운로드
            </Button>

            <Button type="button" variant="secondary" onClick={downloadJSON}>
              JSON 다운로드
            </Button>

            <Button type="button" variant="secondary" onClick={resetLocalLogs}>
              로컬 로그 초기화
            </Button>
          </div>
        </div>
      </Card>

      <Card
        title="실험자용 로그"
        subtitle="태블릿 로컬 로그입니다. 서버에도 별도로 저장을 시도합니다."
      >
        <div className="max-h-[50vh] overflow-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b bg-zinc-100">
                <th className="p-2">Trial</th>
                <th className="p-2">피험자</th>
                <th className="p-2">Session</th>
                <th className="p-2">기준 코드</th>
                <th className="p-2">기준</th>
                <th className="p-2">안내 문장</th>
                <th className="p-2">시간</th>
              </tr>
            </thead>

            <tbody>
              {logs.map((log) => (
                <tr
                  key={`${log.participantId}-${log.sessionId}-${log.trial}-${log.timestamp}`}
                  className="border-b"
                >
                  <td className="p-2 text-center">{log.trial}</td>
                  <td className="p-2 text-center">{log.participantId}</td>
                  <td className="p-2 text-center">{log.sessionId}</td>
                  <td className="p-2 text-center">{log.criterion_code}</td>
                  <td className="p-2 text-center">{log.criterion_label}</td>
                  <td className="p-2 text-center">{log.spoken_instruction}</td>
                  <td className="p-2 text-center">{log.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
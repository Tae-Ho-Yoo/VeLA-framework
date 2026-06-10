import { useEffect, useState } from 'react'
import { Card } from '../components/Card'
import { Button } from '../components/Button'

type ColorItem = {
  code: string
  label: string
  bgClass: string
  textClass: string
}

type ColorOrderLog = {
  trial: number
  order_codes: string[]
  order_labels: string[]
  timestamp: string
}

const COLORS: ColorItem[] = [
  {
    code: 'red',
    label: '빨강',
    bgClass: 'bg-red-500',
    textClass: 'text-white',
  },
  {
    code: 'yellow',
    label: '노랑',
    bgClass: 'bg-yellow-300',
    textClass: 'text-zinc-900',
  },
  {
    code: 'blue',
    label: '파랑',
    bgClass: 'bg-blue-500',
    textClass: 'text-white',
  },
  {
    code: 'black',
    label: '검정',
    bgClass: 'bg-zinc-900',
    textClass: 'text-white',
  },
]

const LOCAL_STORAGE_KEY = 'stroop_color_order_logs'

function shuffleArray<T>(array: T[]) {
  const copied = [...array]

  for (let i = copied.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[copied[i], copied[j]] = [copied[j], copied[i]]
  }

  return copied
}

function loadLocalLogs(): ColorOrderLog[] {
  const saved = localStorage.getItem(LOCAL_STORAGE_KEY)

  if (!saved) return []

  try {
    return JSON.parse(saved)
  } catch {
    return []
  }
}

function saveLocalLogs(logs: ColorOrderLog[]) {
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(logs))
}

function convertLogsToCSV(logs: ColorOrderLog[]) {
  const header = ['trial', 'order_codes', 'order_labels', 'timestamp']

  const rows = logs.map((log) => [
    log.trial,
    log.order_codes.join('|'),
    log.order_labels.join('|'),
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

export function ColorOrderPage() {
  const [currentOrder, setCurrentOrder] = useState<ColorItem[]>(COLORS)
  const [logs, setLogs] = useState<ColorOrderLog[]>([])

  useEffect(() => {
    const savedLogs = loadLocalLogs()
    setLogs(savedLogs)
  }, [])

  function generateRandomOrder() {
    const shuffled = shuffleArray(COLORS)

    const newLog: ColorOrderLog = {
      trial: logs.length + 1,
      order_codes: shuffled.map((color) => color.code),
      order_labels: shuffled.map((color) => color.label),
      timestamp: new Date().toISOString(),
    }

    const updatedLogs = [...logs, newLog]

    setCurrentOrder(shuffled)
    setLogs(updatedLogs)
    saveLocalLogs(updatedLogs)
  }

  function downloadCSV() {
    if (logs.length === 0) {
      alert('저장된 색깔 순서 로그가 없습니다.')
      return
    }

    const csvContent = convertLogsToCSV(logs)
    const blob = new Blob(['\uFEFF' + csvContent], {
      type: 'text/csv;charset=utf-8;',
    })

    downloadBlob(blob, makeFilename('color_order_logs', 'csv'))
  }

  function downloadJSON() {
    if (logs.length === 0) {
      alert('저장된 색깔 순서 로그가 없습니다.')
      return
    }

    const jsonContent = JSON.stringify(logs, null, 2)
    const blob = new Blob([jsonContent], {
      type: 'application/json;charset=utf-8;',
    })

    downloadBlob(blob, makeFilename('color_order_logs', 'json'))
  }

  function resetLogs() {
    const ok = confirm('정말 색깔 순서 로그를 삭제할까요?')

    if (!ok) return

    setLogs([])
    saveLocalLogs([])
    setCurrentOrder(COLORS)
  }

  return (
    <div className="container-page space-y-6">
      <div className="space-y-2">
        <div className="text-2xl font-semibold text-zinc-900">
          색깔 순서 랜덤 제시
        </div>

        <div className="text-sm text-zinc-600">
          버튼을 누르면 빨강, 노랑, 파랑, 검정의 순서가 랜덤으로 지정됩니다.
        </div>
      </div>

      <Card title="현재 색깔 순서" subtitle="왼쪽부터 오른쪽 순서로 사용하세요.">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          {currentOrder.map((color, index) => (
            <div
              key={`${color.code}-${index}`}
              className={`flex h-36 flex-col items-center justify-center rounded-2xl text-3xl font-bold shadow ${color.bgClass} ${color.textClass}`}
            >
              <div className="text-lg opacity-80">{index + 1}</div>
              <div>{color.label}</div>
            </div>
          ))}
        </div>

        <div className="mt-8 flex justify-center">
          <Button
            type="button"
            onClick={generateRandomOrder}
            className="px-12 py-6 text-2xl"
          >
            색깔 순서 랜덤 생성
          </Button>
        </div>

        <div className="mt-4 text-center text-lg text-zinc-600">
          현재 Trial: {logs.length}
        </div>
      </Card>

      <Card title="저장 및 초기화" subtitle="태블릿 브라우저에 자동 저장됩니다.">
        <div className="flex flex-wrap justify-center gap-2">
          <Button type="button" variant="secondary" onClick={downloadCSV}>
            CSV 다운로드
          </Button>

          <Button type="button" variant="secondary" onClick={downloadJSON}>
            JSON 다운로드
          </Button>

          <Button type="button" variant="secondary" onClick={resetLogs}>
            로그 초기화
          </Button>
        </div>
      </Card>

      <Card title="색깔 순서 로그">
        <div className="max-h-[45vh] overflow-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b bg-zinc-100">
                <th className="p-2">Trial</th>
                <th className="p-2">순서 코드</th>
                <th className="p-2">순서 이름</th>
                <th className="p-2">시간</th>
              </tr>
            </thead>

            <tbody>
              {logs.map((log) => (
                <tr key={`${log.trial}-${log.timestamp}`} className="border-b">
                  <td className="p-2 text-center">{log.trial}</td>
                  <td className="p-2 text-center">
                    {log.order_codes.join(' → ')}
                  </td>
                  <td className="p-2 text-center">
                    {log.order_labels.join(' → ')}
                  </td>
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
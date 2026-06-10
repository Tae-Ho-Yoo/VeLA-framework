import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import axios from 'axios'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DATA_DIR = path.join(__dirname, 'data')
const PORT = Number(process.env.PORT) || 3001

const STROOP_JSON_PATH = path.join(DATA_DIR, 'stroop_instruction_logs.json')
const STROOP_CSV_PATH = path.join(DATA_DIR, 'stroop_instruction_logs.csv')

const app = express()
app.use(express.json({ limit: '5mb' }))

function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true })
  }
}

function safeFilenamePart(s) {
  return String(s ?? 'unknown').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 80)
}

function extractValence(record) {
  return (
    record.sam?.valence ??
    record.responses?.valence ??
    record.responses?.sam?.valence ??
    record.valence ??
    null
  )
}

function extractArousal(record) {
  return (
    record.sam?.arousal ??
    record.responses?.arousal ??
    record.responses?.sam?.arousal ??
    record.arousal ??
    null
  )
}

function extractDominance(record) {
  return (
    record.sam?.dominance ??
    record.responses?.dominance ??
    record.responses?.sam?.dominance ??
    record.dominance ??
    null
  )
}

function vadToEmotion(valence, arousal, dominance) {
  if (valence == null || arousal == null || dominance == null) {
    return 'neutral'
  }

  const v = Number(valence)
  const a = Number(arousal)
  const d = Number(dominance)

  if (Number.isNaN(v) || Number.isNaN(a) || Number.isNaN(d)) {
    return 'neutral'
  }

  // 1차 규칙: valence 중심
  if (v >= 6) return 'positive'
  if (v <= 3) return 'negative'
  return 'neutral'
}

function readAllJsonl() {
  const file = path.join(DATA_DIR, 'all.jsonl')
  if (!fs.existsSync(file)) return []

  const raw = fs.readFileSync(file, 'utf8').trim()
  if (!raw) return []

  return raw
    .split('\n')
    .map(function (line) {
      try {
        return JSON.parse(line)
      } catch {
        return null
      }
    })
    .filter(Boolean)
}

function buildEmotionHistory(userId) {
  const data = readAllJsonl()

  return data
    .filter(function (d) {
      return d.participant && d.participant.participant_id === userId
    })
    .map(function (d) {
      return d.emotion || 'neutral'
    })
    .slice(-3)
}

async function callPythonAct(payload) {
  const res = await axios.post('http://127.0.0.1:8000/act', payload, {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 0,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  })

  return res.data
}

function csvEscape(value) {
  const escaped = String(value ?? '').replaceAll('"', '""')
  return `"${escaped}"`
}

function readJsonArray(filePath) {
  if (!fs.existsSync(filePath)) {
    return []
  }

  try {
    const raw = fs.readFileSync(filePath, 'utf8')
    if (!raw.trim()) return []
    return JSON.parse(raw)
  } catch (error) {
    console.error('JSON read error:', error)
    return []
  }
}

function writeJsonArray(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8')
}

function ensureStroopCsvHeader() {
  ensureDataDir()

  if (!fs.existsSync(STROOP_CSV_PATH)) {
    const header = [
      'trial',
      'participantId',
      'sessionId',
      'criterion_code',
      'criterion_label',
      'spoken_instruction',
      'timestamp',
    ].join(',')

    fs.writeFileSync(STROOP_CSV_PATH, header + '\n', 'utf8')
  }
}

// NASA-TLX / survey 저장 API
app.post('/api/submissions', async function (req, res) {
  const record = req.body

  if (!record || typeof record !== 'object') {
    return res.status(400).json({ ok: false, error: 'invalid body' })
  }

  ensureDataDir()

  const userId =
    (record.participant && record.participant.participant_id) || 'unknown_user'

  const valence = extractValence(record)
  const arousal = extractArousal(record)
  const dominance = extractDominance(record)
  const emotion = vadToEmotion(valence, arousal, dominance)

  // 기존 sam은 유지하고, 정규화된 값은 별도로 저장
  record.vad = {
    valence: valence,
    arousal: arousal,
    dominance: dominance,
  }
  record.emotion = emotion

  console.log('VAD:', {
    valence: valence,
    arousal: arousal,
    dominance: dominance,
    emotion: emotion,
  })

  const ts = safeFilenamePart(record.created_at || Date.now())
  const fname = safeFilenamePart(userId) + '_' + ts + '.json'
  const fpath = path.join(DATA_DIR, fname)

  try {
    const existed = fs.existsSync(fpath)
    fs.writeFileSync(fpath, JSON.stringify(record, null, 2), 'utf8')

    const jsonl = path.join(DATA_DIR, 'all.jsonl')
    if (!existed) {
      fs.appendFileSync(jsonl, JSON.stringify(record) + '\n', 'utf8')
    }

    const emotion_history = buildEmotionHistory(userId)
    console.log('emotion_history:', emotion_history)

    let robotResult = null

    try {
      robotResult = await callPythonAct({
        instruction: 'give me the cup',
        user_id: userId,
        emotion_history: emotion_history,
        unnorm_key: 'bridge_orig',
      })

      console.log('robot result:', robotResult)

      const robotLogPath = path.join(DATA_DIR, 'robot_logs.jsonl')
      fs.appendFileSync(
        robotLogPath,
        JSON.stringify({
          user_id: userId,
          vad: {
            valence: valence,
            arousal: arousal,
            dominance: dominance,
          },
          emotion: emotion,
          emotion_history: emotion_history,
          result: robotResult,
          created_at: new Date().toISOString(),
        }) + '\n',
        'utf8'
      )
    } catch (e) {
      console.error('Python call failed:', e)
    }

    return res.json({
      ok: true,
      file: fname,
      vad: {
        valence: valence,
        arousal: arousal,
        dominance: dominance,
      },
      emotion: emotion,
      emotion_history: emotion_history,
      robot_result: robotResult,
    })
  } catch (e) {
    console.error(e)
    return res.status(500).json({ ok: false, error: 'write failed' })
  }
})

// 실험자용 Stroop 기준 안내 로그 저장 API
app.post('/api/stroop-log', function (req, res) {
  const log = req.body

  if (!log || typeof log !== 'object') {
    return res.status(400).json({
      ok: false,
      error: 'invalid body',
    })
  }

  const requiredFields = [
    'trial',
    'participantId',
    'sessionId',
    'criterion_code',
    'criterion_label',
    'spoken_instruction',
    'timestamp',
  ]

  for (const field of requiredFields) {
    if (log[field] === undefined || log[field] === null || log[field] === '') {
      return res.status(400).json({
        ok: false,
        error: `Missing field: ${field}`,
      })
    }
  }

  ensureDataDir()

  try {
    const logs = readJsonArray(STROOP_JSON_PATH)
    logs.push(log)
    writeJsonArray(STROOP_JSON_PATH, logs)

    ensureStroopCsvHeader()

    const row = [
      log.trial,
      log.participantId,
      log.sessionId,
      log.criterion_code,
      log.criterion_label,
      log.spoken_instruction,
      log.timestamp,
    ]
      .map(csvEscape)
      .join(',')

    fs.appendFileSync(STROOP_CSV_PATH, row + '\n', 'utf8')

    console.log('Stroop log saved:', {
      trial: log.trial,
      participantId: log.participantId,
      sessionId: log.sessionId,
      criterion: log.criterion_code,
    })

    return res.json({
      ok: true,
      saved: log,
      total: logs.length,
    })
  } catch (e) {
    console.error('Stroop log write failed:', e)

    return res.status(500).json({
      ok: false,
      error: 'stroop log write failed',
    })
  }
})

// 실험자용 Stroop 로그 확인 API
app.get('/api/stroop-logs', function (_req, res) {
  ensureDataDir()

  const logs = readJsonArray(STROOP_JSON_PATH)

  return res.json({
    ok: true,
    total: logs.length,
    logs,
  })
})

app.get('/api/health', function (_req, res) {
  res.json({ ok: true })
})

app.listen(PORT, function () {
  console.log('[vela-survey-api] http://localhost:' + PORT)
  console.log('[vela-survey-api] POST /api/submissions -> ' + DATA_DIR)
  console.log('[vela-survey-api] POST /api/stroop-log -> ' + STROOP_CSV_PATH)
})
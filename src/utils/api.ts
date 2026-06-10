const base = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

export type SubmitResult = { ok: true; file?: string } | { ok: false; error: string }

export async function submitRecord(record: unknown): Promise<SubmitResult> {
  try {
    const url = `${base}/api/submissions`
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record),
    })
    const text = await r.text()
    let data: { ok?: boolean; file?: string; error?: string } = {}
    try {
      data = JSON.parse(text) as typeof data
    } catch {
      if (!r.ok) return { ok: false, error: text || r.statusText }
    }
    if (!r.ok) return { ok: false, error: data.error || text || r.statusText }
    return { ok: true, file: data.file }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

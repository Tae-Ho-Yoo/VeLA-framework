import { useId } from 'react'
import { cn } from './cn'

const STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9] as const

export function SamSlider({
  label,
  description,
  value,
  onChange,
  leftLabel = '낮음',
  rightLabel = '높음',
}: {
  label: string
  description?: string
  value: number
  onChange: (next: number) => void
  leftLabel?: string
  rightLabel?: string
}) {
  const id = useId()
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="text-base font-medium text-zinc-900">{label}</div>
          {description ? <div className="text-sm text-zinc-600">{description}</div> : null}
        </div>
        <div className="shrink-0 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-center shadow-sm">
          <div className="text-xs text-zinc-500">선택</div>
          <div className="text-lg font-bold tabular-nums text-zinc-900">{value}</div>
        </div>
      </div>

      <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4">
        <div className="-mx-1 overflow-x-auto pb-1 sm:mx-0 sm:overflow-visible">
          <div className="grid min-w-[20rem] grid-cols-9 gap-2 sm:min-w-0">
          {STEPS.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onChange(n)}
              className={cn(
                'flex min-h-10 items-center justify-center rounded-md border text-sm font-semibold tabular-nums transition sm:min-h-12 sm:text-base',
                value === n
                  ? 'border-zinc-900 bg-zinc-900 text-white shadow-sm'
                  : 'border-zinc-300 bg-white text-zinc-800 hover:border-zinc-400 hover:bg-zinc-100',
              )}
              aria-pressed={value === n}
              aria-label={`${n}점`}
            >
              {n}
            </button>
          ))}
          </div>
        </div>

        <div className="mt-4 px-0.5">
          <input
            id={id}
            type="range"
            min={1}
            max={9}
            step={1}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className={cn('h-3 w-full cursor-pointer accent-zinc-900')}
            aria-label={label}
          />
        </div>

        <div className="mt-3 flex items-start justify-between gap-2 text-xs text-zinc-600 sm:text-sm">
          <span className="max-w-[40%] text-left leading-snug">{leftLabel}</span>
          <span className="max-w-[40%] text-right leading-snug">{rightLabel}</span>
        </div>
      </div>
    </div>
  )
}


import type { ReactNode } from 'react'

export function Card({
  title,
  subtitle,
  right,
  children,
}: {
  title: string
  subtitle?: string
  right?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="rounded-xl bg-white shadow-sm ring-1 ring-zinc-200">
      <div className="flex items-start justify-between gap-4 border-b border-zinc-100 px-6 py-4">
        <div className="space-y-1">
          <div className="text-base font-semibold text-zinc-900">{title}</div>
          {subtitle ? <div className="text-sm text-zinc-500">{subtitle}</div> : null}
        </div>
        {right}
      </div>
      <div className="px-6 py-5">{children}</div>
    </div>
  )
}


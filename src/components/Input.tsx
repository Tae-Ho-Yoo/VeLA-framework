import type { InputHTMLAttributes } from 'react'
import { cn } from './cn'

type Props = InputHTMLAttributes<HTMLInputElement> & {
  hasError?: boolean
}

export function Input({ className, hasError, ...props }: Props) {
  return (
    <input
      className={cn(
        'w-full rounded-md border bg-white px-3 py-2 text-sm outline-none transition placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-2 focus:ring-zinc-200',
        hasError ? 'border-red-400 focus:border-red-400 focus:ring-red-100' : 'border-zinc-200',
        className,
      )}
      {...props}
    />
  )
}


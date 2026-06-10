type Props = {
  label: string
  description?: string
  value: number
  onChange: (value: number) => void
  leftLabel?: string
  rightLabel?: string
}

export function TLXRadio({
  label,
  description,
  value,
  onChange,
  leftLabel,
  rightLabel,
}: Props) {
  const options = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-xl font-bold">{label}</h2>
        {description && (
          <p className="mt-1 text-sm text-zinc-600">{description}</p>
        )}
      </div>

      <div className="mx-auto w-fit">
        <div className="mb-2 flex justify-between text-sm text-zinc-500">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>

        <div className="grid grid-cols-9 gap-3">
          {options.map((n) => (
            <label
              key={n}
              className={`
                flex h-10 w-10 items-center justify-center
                rounded-full border cursor-pointer transition
                ${
                  value === n
                    ? 'border-black bg-black text-white'
                    : 'border-zinc-300 bg-white text-black hover:border-black'
                }
              `}
            >
              <input
                type="radio"
                className="sr-only"
                checked={value === n}
                onChange={() => onChange(n)}
              />
              {n}
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
interface ChipProps {
  label: string
  onClick: () => void
  color?: 'raman' | 'sers'
}

export default function Chip({ label, onClick, color = 'raman' }: ChipProps) {
  const colorClasses = color === 'raman'
    ? 'bg-raman-50 text-raman-700 border-raman-200 hover:bg-raman-100'
    : 'bg-sers-50 text-sers-700 border-sers-200 hover:bg-sers-100'

  return (
    <button
      onClick={onClick}
      className={`
        px-5 py-2.5 rounded-full text-sm font-medium
        border transition-all
        focus:ring-2 focus:ring-offset-2 focus:ring-raman-500
        ${colorClasses}
      `}
    >
      {label}
    </button>
  )
}


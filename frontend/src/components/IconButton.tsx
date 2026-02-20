import { LucideIcon } from 'lucide-react'

interface IconButtonProps {
  icon: LucideIcon
  onClick: () => void
  label: string
  className?: string
}

export default function IconButton({ icon: Icon, onClick, label, className = '' }: IconButtonProps) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className={`
        p-2 hover:bg-gray-100 rounded-lg
        focus:ring-2 focus:ring-raman-500
        transition-colors
        ${className}
      `}
    >
      <Icon className="w-5 h-5 text-gray-700" />
    </button>
  )
}


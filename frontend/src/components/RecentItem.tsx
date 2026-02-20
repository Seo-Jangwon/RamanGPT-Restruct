import { MessageSquare } from 'lucide-react'

interface RecentItemProps {
  title: string
}

export default function RecentItem({ title }: RecentItemProps) {
  return (
    <button
      className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-gray-700 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500 group"
      aria-label={`Open chat: ${title}`}
    >
      <MessageSquare className="w-4 h-4 text-gray-400 group-hover:text-gray-600 flex-shrink-0" />
      <span className="truncate">{title}</span>
    </button>
  )
}


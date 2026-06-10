import { FlaskConical, LogOut } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { useSurveyStore } from '../store/surveyStore'
import { Button } from './Button'

export function TopBar() {
  const navigate = useNavigate()
  const logout = useSurveyStore((s) => s.logout)
  const participant = useSurveyStore((s) => s.participant)
  const hasParticipant = Boolean(participant)

  const homeTo = hasParticipant ? '/survey' : '/'

  const handleLogout = () => {
    logout()
    navigate('/', { replace: true })
  }

  return (
    <div className="border-b border-zinc-200 bg-white">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3">
        <Link to={homeTo} className="inline-flex items-center gap-2 text-sm font-semibold text-zinc-900">
          <FlaskConical className="h-4 w-4" />
          VELA Survey
        </Link>
        <div className="flex items-center gap-2">
          {hasParticipant ? (
            <>
              <span className="hidden max-w-[10rem] truncate text-xs text-zinc-500 sm:inline" title={participant?.name}>
                {participant?.name}
              </span>
              <Button
                type="button"
                variant="secondary"
                onClick={handleLogout}
                className="gap-2"
                title="로그아웃 후 참가 정보를 다시 입력합니다"
              >
                <LogOut className="h-4 w-4" />
                로그아웃
              </Button>
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}


import { useNavigate } from 'react-router-dom'
import { Card } from '../components/Card'
import { Button } from '../components/Button'

export function NotFoundPage() {
  const navigate = useNavigate()
  return (
    <div className="container-page">
      <Card title="404" subtitle="페이지를 찾을 수 없습니다.">
        <div className="flex items-center justify-end">
          <Button type="button" onClick={() => navigate('/')}>
            인트로로 이동
          </Button>
        </div>
      </Card>
    </div>
  )
}


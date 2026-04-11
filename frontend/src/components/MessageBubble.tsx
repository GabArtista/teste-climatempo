/** Renders a single chat message bubble with optional weather card. */

import type { Message } from '../types'
import { WeatherCard } from './WeatherCard'

interface Props {
  message: Message
}

const isUser = (role: string) => role === 'user'

export function MessageBubble({ message }: Props) {
  const user = isUser(message.role)

  const bubbleStyle: React.CSSProperties = {
    maxWidth: '78%',
    padding: '10px 14px',
    borderRadius: user ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
    background: user ? '#2563eb' : '#1e293b',
    color: user ? '#fff' : '#e2e8f0',
    fontSize: 15,
    lineHeight: 1.6,
    wordBreak: 'break-word',
    border: user ? 'none' : '1px solid #334155',
  }

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: user ? 'flex-end' : 'flex-start',
    gap: 4,
    marginBottom: 16,
  }

  const metaStyle: React.CSSProperties = {
    fontSize: 11,
    color: '#64748b',
    paddingLeft: user ? 0 : 4,
    paddingRight: user ? 4 : 0,
  }

  const timeStr = message.timestamp.toLocaleTimeString('pt-BR', {
    hour: '2-digit',
    minute: '2-digit',
  })

  // Split text: lines before forecast card + forecast lines
  const hasWeather = message.toolCalled && message.content.includes('📅')
  const introText = hasWeather
    ? message.content.split('\n').filter(l => !l.includes('📅') && !l.includes('🌤️')).join('\n').trim()
    : message.content

  return (
    <div style={containerStyle}>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, flexDirection: user ? 'row-reverse' : 'row' }}>
        {/* Avatar */}
        <div style={{
          width: 32, height: 32, borderRadius: '50%',
          background: user ? '#3b82f6' : '#334155',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 16, flexShrink: 0,
        }}>
          {user ? '👤' : '🌦️'}
        </div>

        <div style={{ maxWidth: '100%' }}>
          {introText && <div style={bubbleStyle}>{introText}</div>}
          {hasWeather && <WeatherCard text={message.content} />}
        </div>
      </div>

      <div style={metaStyle}>
        {timeStr}
        {message.toolCalled && (
          <span style={{ marginLeft: 6, color: '#22c55e' }}>
            ⚡ Open-Meteo
          </span>
        )}
      </div>
    </div>
  )
}

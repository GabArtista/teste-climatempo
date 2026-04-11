/** Parses and renders a structured weather forecast from agent response text. */

interface Props {
  text: string
}

interface ParsedDay {
  date: string
  line: string
}

function parseWeatherLines(text: string): ParsedDay[] {
  return text
    .split('\n')
    .filter(l => l.includes('📅'))
    .map(l => ({ date: l.split(':')[0].replace('📅', '').trim(), line: l.trim() }))
}

const cardStyle: React.CSSProperties = {
  background: 'linear-gradient(135deg, #1e3a5f 0%, #0f2640 100%)',
  border: '1px solid #2d5a8e',
  borderRadius: 12,
  padding: '16px 20px',
  marginTop: 8,
}

const dayStyle: React.CSSProperties = {
  padding: '8px 0',
  borderBottom: '1px solid #1e3a5f',
  fontSize: 14,
  color: '#cbd5e1',
  lineHeight: 1.6,
}

export function WeatherCard({ text }: Props) {
  const lines = parseWeatherLines(text)
  if (lines.length === 0) return null

  // Extract city from first line (🌤️ Previsão do tempo para CITY:)
  const cityLine = text.split('\n').find(l => l.includes('🌤️'))
  const city = cityLine?.replace('🌤️ Previsão do tempo para', '').replace(':', '').trim()

  return (
    <div style={cardStyle}>
      {city && (
        <div style={{ fontSize: 16, fontWeight: 600, color: '#7dd3fc', marginBottom: 12 }}>
          🌤️ {city}
        </div>
      )}
      {lines.map((d, i) => (
        <div
          key={i}
          style={{ ...dayStyle, borderBottom: i < lines.length - 1 ? dayStyle.borderBottom : 'none' }}
        >
          {d.line}
        </div>
      ))}
    </div>
  )
}

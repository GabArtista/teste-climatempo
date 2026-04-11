/** Domain types shared across the frontend. */

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  toolCalled?: boolean
  cityQueried?: string | null
  timestamp: Date
}

export interface DailyForecast {
  date: string
  temp_max: number
  temp_min: number
  precipitation: number
}

export interface WeatherData {
  city: string
  latitude: number
  longitude: number
  forecasts: DailyForecast[]
  generated_at: string
}

export interface ChatRequest {
  message: string
  history: Array<{
    role: string
    content: string
  }>
}

export interface ChatResponse {
  response: string
  tool_called: boolean
  city_queried: string | null
}

export type ChatStatus = 'idle' | 'loading' | 'error'

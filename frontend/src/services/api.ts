/** API client for the Weather LLM Agent backend. */

import type { ChatRequest, ChatResponse, WeatherData } from '../types'

const BASE_URL = '/api/v1'

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

/** Send a message to the LLM agent and get a response. */
export async function sendChat(payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(90_000),
  })
  return handleResponse<ChatResponse>(res)
}

/** Fetch weather directly (bypassing LLM) for a city and number of days. */
export async function getWeather(city: string, days = 3): Promise<WeatherData> {
  const params = new URLSearchParams({ city, days: String(days) })
  const res = await fetch(`${BASE_URL}/weather/?${params}`, {
    signal: AbortSignal.timeout(15_000),
  })
  return handleResponse<WeatherData>(res)
}

/** Check if the backend and Ollama are healthy. */
export async function checkHealth(): Promise<{ status: string; model: string; available: boolean }> {
  const res = await fetch(`${BASE_URL}/agent/health`, {
    signal: AbortSignal.timeout(5_000),
  })
  return handleResponse(res)
}

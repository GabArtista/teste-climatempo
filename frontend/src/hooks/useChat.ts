/** Hook managing the full chat conversation state. */

import { useCallback, useRef, useState } from 'react'
import { sendChat } from '../services/api'
import type { ChatStatus, Message } from '../types'

function makeId(): string {
  return Math.random().toString(36).slice(2)
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<ChatStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // Keep a ref of messages for building history without stale closure
  const messagesRef = useRef<Message[]>([])
  messagesRef.current = messages

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || status === 'loading') return

    const userMsg: Message = {
      id: makeId(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMsg])
    setStatus('loading')
    setError(null)

    // Build history from current messages (excluding the one we just added)
    const history = messagesRef.current.map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      const response = await sendChat({ message: text.trim(), history })

      const assistantMsg: Message = {
        id: makeId(),
        role: 'assistant',
        content: response.response,
        toolCalled: response.tool_called,
        cityQueried: response.city_queried,
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, assistantMsg])
      setStatus('idle')
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Erro ao conectar com o servidor.'
      setError(message)
      setStatus('error')
    }
  }, [status])

  const clearChat = useCallback(() => {
    setMessages([])
    setStatus('idle')
    setError(null)
  }, [])

  return { messages, status, error, sendMessage, clearChat }
}

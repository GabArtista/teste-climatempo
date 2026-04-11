/** Main chat interface — input, message list, status bar. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { LoadingDots } from './LoadingDots'
import { MessageBubble } from './MessageBubble'
import { useChat } from '../hooks/useChat'

const SUGGESTIONS = [
  'Como está o tempo em São Paulo hoje?',
  'Previsão para Curitiba nos próximos 5 dias',
  'Vai chover em Manaus amanhã?',
  'Temperatura mínima em Brasília esta semana',
]

export function ChatWindow() {
  const { messages, status, error, sendMessage, clearChat } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  const handleSend = useCallback(() => {
    if (!input.trim()) return
    sendMessage(input)
    setInput('')
    inputRef.current?.focus()
  }, [input, sendMessage])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div style={styles.shell}>
      {/* Header */}
      <header style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 24 }}>🌦️</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>Weather Agent</div>
            <div style={{ fontSize: 12, color: '#64748b' }}>
              Powered by Ollama · qwen2.5:1.5b · Open-Meteo
            </div>
          </div>
        </div>
        {messages.length > 0 && (
          <button onClick={clearChat} style={styles.clearBtn} title="Limpar conversa">
            🗑
          </button>
        )}
      </header>

      {/* Messages */}
      <div style={styles.messages}>
        {isEmpty ? (
          <div style={styles.empty}>
            <div style={{ fontSize: 56, marginBottom: 16 }}>⛅</div>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
              Pergunte sobre o tempo
            </div>
            <div style={{ fontSize: 14, color: '#64748b', marginBottom: 24 }}>
              Previsão diária para qualquer capital brasileira
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', maxWidth: 360 }}>
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  style={styles.suggestionBtn}
                  onClick={() => { sendMessage(s); inputRef.current?.focus() }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {status === 'loading' && (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 16 }}>
                <div style={{ width: 32, height: 32, borderRadius: '50%', background: '#334155', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 }}>
                  🌦️
                </div>
                <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '18px 18px 18px 4px' }}>
                  <LoadingDots />
                </div>
              </div>
            )}
            {error && (
              <div style={styles.errorBanner}>
                ⚠️ {error}
              </div>
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={styles.inputArea}>
        <input
          ref={inputRef}
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ex: Como está o tempo em Fortaleza?"
          disabled={status === 'loading'}
          autoFocus
        />
        <button
          style={{
            ...styles.sendBtn,
            opacity: !input.trim() || status === 'loading' ? 0.4 : 1,
            cursor: !input.trim() || status === 'loading' ? 'not-allowed' : 'pointer',
          }}
          onClick={handleSend}
          disabled={!input.trim() || status === 'loading'}
          title="Enviar (Enter)"
        >
          ➤
        </button>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  shell: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    maxWidth: 760,
    margin: '0 auto',
    background: '#0f172a',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 20px',
    borderBottom: '1px solid #1e293b',
    background: '#0f172a',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  clearBtn: {
    background: 'none',
    border: 'none',
    color: '#64748b',
    fontSize: 18,
    cursor: 'pointer',
    padding: 4,
    borderRadius: 6,
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px 20px 8px',
  },
  empty: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    textAlign: 'center',
    color: '#94a3b8',
    minHeight: 400,
  },
  suggestionBtn: {
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 10,
    color: '#cbd5e1',
    padding: '10px 14px',
    cursor: 'pointer',
    textAlign: 'left',
    fontSize: 14,
    transition: 'background 0.15s',
  },
  inputArea: {
    display: 'flex',
    gap: 8,
    padding: '12px 16px',
    borderTop: '1px solid #1e293b',
    background: '#0f172a',
  },
  input: {
    flex: 1,
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 12,
    padding: '10px 16px',
    color: '#f1f5f9',
    fontSize: 15,
    outline: 'none',
  },
  sendBtn: {
    background: '#2563eb',
    border: 'none',
    borderRadius: 12,
    color: '#fff',
    fontSize: 18,
    width: 44,
    cursor: 'pointer',
    flexShrink: 0,
  },
  errorBanner: {
    background: '#7f1d1d',
    border: '1px solid #991b1b',
    borderRadius: 8,
    padding: '10px 14px',
    fontSize: 14,
    color: '#fca5a5',
    marginBottom: 12,
  },
}

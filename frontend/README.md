# Weather Agent — Frontend

Interface React para o agente de previsão do tempo.

## Stack
- React 18 + TypeScript
- Vite (dev server + build)
- Proxy para o backend FastAPI em `/api`

## Instalação e execução

```bash
# Instalar dependências
npm install

# Desenvolvimento (com hot reload)
npm run dev
# Acesse: http://localhost:5173

# Build de produção
npm run build
npm run preview
```

## Variáveis de ambiente

O frontend usa o proxy do Vite para rotear `/api/*` → `http://localhost:8000`.
Para produção, ajuste `vite.config.ts` com a URL correta do backend.

## Estrutura

```
src/
├── types/index.ts         # Tipos TypeScript compartilhados
├── services/api.ts        # Cliente HTTP para o backend
├── hooks/useChat.ts       # Estado e lógica da conversa
└── components/
    ├── ChatWindow.tsx     # Container principal do chat
    ├── MessageBubble.tsx  # Renderiza mensagem individual
    ├── WeatherCard.tsx    # Card visual da previsão
    └── LoadingDots.tsx    # Indicador de carregamento
```

## Pré-requisito

O backend deve estar rodando em `http://localhost:8000`:

```bash
cd ../backend
uvicorn main:app --port 8000
```

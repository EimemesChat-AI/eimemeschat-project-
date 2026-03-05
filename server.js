/*
  EimemesChat AI — Express backend
  Multi-model failover: Claude → GPT-4o → Gemini
  Each provider has a hard 25 s timeout.
  If all fail the client gets a clean 503.
*/

'use strict';
require('dotenv').config();

const express = require('express');
const path    = require('path');
const app     = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Logger ──────────────────────────────────────────────────────
const ts  = () => new Date().toISOString();
const log = (tag, msg) => console.log(`[${ts()}] [${tag}] ${msg}`);

// ── Per-provider timeout helper ─────────────────────────────────
// Wraps a fetch promise and rejects after `ms` milliseconds.
function withTimeout(promise, ms, label) {
  let id;
  const timer = new Promise((_, reject) => {
    id = setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms);
  });
  return Promise.race([promise, timer]).finally(() => clearTimeout(id));
}

const PROVIDER_TIMEOUT = 25_000; // 25 s per provider

// ── AI providers ────────────────────────────────────────────────

async function tryClaude(messages, abortSignal) {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error('ANTHROPIC_API_KEY not set');

  const fetchCall = fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    signal: abortSignal,
    headers: {
      'Content-Type':    'application/json',
      'x-api-key':       process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      // claude-sonnet-4-6 is faster + cheaper than Opus; swap to opus-4-6 if you need max quality
      model:      'claude-sonnet-4-6',
      max_tokens: 1024,
      system:     'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted. Be concise unless detail is requested.',
      messages:   messages.map(m => ({
        role:    m.role === 'assistant' ? 'assistant' : 'user',
        content: m.content,
      })),
    }),
  });

  const res = await withTimeout(fetchCall, PROVIDER_TIMEOUT, 'Claude');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Claude HTTP ${res.status}: ${body.slice(0, 160)}`);
  }
  const data = await res.json();
  if (!data?.content?.[0]?.text) throw new Error('Claude: unexpected response shape');
  return { reply: data.content[0].text, model: 'Claude' };
}

async function tryOpenAI(messages, abortSignal) {
  if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY not set');

  const fetchCall = fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    signal: abortSignal,
    headers: {
      'Content-Type': 'application/json',
      Authorization:  `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',   // gpt-4o-mini is faster; change to gpt-4o for higher quality
      max_tokens: 1024,
      messages: [
        {
          role:    'system',
          content: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted.',
        },
        ...messages.map(m => ({
          role:    m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content,
        })),
      ],
    }),
  });

  const res = await withTimeout(fetchCall, PROVIDER_TIMEOUT, 'OpenAI');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`OpenAI HTTP ${res.status}: ${body.slice(0, 160)}`);
  }
  const data = await res.json();
  if (!data?.choices?.[0]?.message?.content) throw new Error('OpenAI: unexpected response shape');
  return { reply: data.choices[0].message.content, model: 'GPT-4o' };
}

async function tryGemini(messages, abortSignal) {
  if (!process.env.GEMINI_API_KEY) throw new Error('GEMINI_API_KEY not set');

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const fetchCall = fetch(url, {
    method: 'POST',
    signal: abortSignal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: {
        parts: [{ text: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant.' }],
      },
      contents: messages.map(m => ({
        role:  m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      })),
      generationConfig: { maxOutputTokens: 1024 },
    }),
  });

  const res = await withTimeout(fetchCall, PROVIDER_TIMEOUT, 'Gemini');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Gemini HTTP ${res.status}: ${body.slice(0, 160)}`);
  }
  const data = await res.json();
  if (!data?.candidates?.[0]?.content?.parts?.[0]?.text)
    throw new Error('Gemini: unexpected response shape');
  return { reply: data.candidates[0].content.parts[0].text, model: 'Gemini' };
}

// Provider chain — first one with a key configured wins
const PROVIDERS = [
  { name: 'Claude', fn: tryClaude  },
  { name: 'OpenAI', fn: tryOpenAI  },
  { name: 'Gemini', fn: tryGemini  },
];

// ── POST /api/chat ───────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  const { message, history = [] } = req.body;

  if (!message?.trim()) {
    return res.status(400).json({ error: 'Message is required.' });
  }

  // Build message array: history + current user message
  const messages = [
    ...history
      .filter(m => m.role && m.content)   // skip malformed entries
      .map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message.trim() },
  ];

  // Abort signal forwarded to fetch calls when client disconnects
  const ctrl = new AbortController();
  req.on('close', () => {
    if (!res.headersSent) ctrl.abort();
  });

  const errors = [];

  for (const { name, fn } of PROVIDERS) {
    if (ctrl.signal.aborted) break; // client already gone

    try {
      log('AI', `Trying ${name}…`);
      const result = await fn(messages, ctrl.signal);
      log('AI', `✓ ${name} responded (${result.reply.length} chars)`);
      return res.json(result);
    } catch (err) {
      if (err.name === 'AbortError' || ctrl.signal.aborted) {
        log('AI', 'Client disconnected — aborting');
        return; // don't send anything; connection is gone
      }
      log('AI', `✗ ${name} failed — ${err.message}`);
      errors.push(`${name}: ${err.message}`);
      // Continue to next provider
    }
  }

  if (!res.headersSent) {
    return res.status(503).json({
      error: 'All AI providers are unavailable. Please try again in a moment.',
      details: errors,
    });
  }
});

// ── GET /api/health ──────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  const envKeys = {
    Claude: 'ANTHROPIC_API_KEY',
    OpenAI: 'OPENAI_API_KEY',
    Gemini: 'GEMINI_API_KEY',
  };
  const configured = Object.entries(envKeys)
    .filter(([, k]) => !!process.env[k])
    .map(([name]) => name);

  res.json({
    status:    'ok',
    providers: configured,
    timestamp: new Date().toISOString(),
  });
});

// ── Catch-all: serve index.html for any unknown route ───────────
// (supports browser refresh on any path)
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start ────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  log('SERVER', `EimemesChat running → http://localhost:${PORT}`);

  const configured = PROVIDERS.filter(p => {
    const keyMap = { Claude: 'ANTHROPIC_API_KEY', OpenAI: 'OPENAI_API_KEY', Gemini: 'GEMINI_API_KEY' };
    return !!process.env[keyMap[p.name]];
  }).map(p => p.name);

  if (configured.length) {
    log('SERVER', `Active providers: ${configured.join(' → ')}`);
  } else {
    log('SERVER', '⚠️  No AI providers configured — add keys to .env');
  }
});

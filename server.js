require('dotenv').config();
const express = require('express');
const path = require('path');

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Logging helper ────────────────────────────────────────────────────────────
const log = (tag, msg) => console.log(`[${new Date().toISOString()}] [${tag}] ${msg}`);

// ── AI Providers ──────────────────────────────────────────────────────────────

async function tryClaude(messages, signal) {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error('No ANTHROPIC_API_KEY');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-opus-4-5',
      max_tokens: 1024,
      system: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted.',
      messages: messages.map(m => ({
        role: m.role === 'assistant' ? 'assistant' : 'user',
        content: m.content,
      })),
    }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => '');
    throw new Error(`Claude HTTP ${res.status}: ${err.slice(0, 120)}`);
  }
  const d = await res.json();
  return { reply: d.content[0].text, model: 'Claude' };
}

async function tryOpenAI(messages, signal) {
  if (!process.env.OPENAI_API_KEY) throw new Error('No OPENAI_API_KEY');
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted.' },
        ...messages.map(m => ({
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content,
        })),
      ],
    }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => '');
    throw new Error(`OpenAI HTTP ${res.status}: ${err.slice(0, 120)}`);
  }
  const d = await res.json();
  return { reply: d.choices[0].message.content, model: 'GPT-4o' };
}

async function tryGemini(messages, signal) {
  if (!process.env.GEMINI_API_KEY) throw new Error('No GEMINI_API_KEY');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant.' }] },
      contents: messages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      })),
    }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => '');
    throw new Error(`Gemini HTTP ${res.status}: ${err.slice(0, 120)}`);
  }
  const d = await res.json();
  return { reply: d.candidates[0].content.parts[0].text, model: 'Gemini' };
}

// Provider chain: order matters — first configured one wins
const PROVIDERS = [
  { name: 'Claude',  fn: tryClaude  },
  { name: 'GPT-4o',  fn: tryOpenAI  },
  { name: 'Gemini',  fn: tryGemini  },
];

// ── Chat endpoint ─────────────────────────────────────────────────────────────

app.post('/api/chat', async (req, res) => {
  const { message, history = [] } = req.body;
  if (!message?.trim()) {
    return res.status(400).json({ error: 'Message is required.' });
  }

  const messages = [
    ...history.map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message },
  ];

  // Abort if client disconnects
  const ctrl = new AbortController();
  req.on('close', () => ctrl.abort());

  const errors = [];

  for (const { name, fn } of PROVIDERS) {
    try {
      log('AI', `Trying ${name}…`);
      const result = await fn(messages, ctrl.signal);
      log('AI', `✓ ${name} responded`);
      return res.json(result);
    } catch (err) {
      if (err.name === 'AbortError') return; // client disconnected
      log('AI', `✗ ${name} failed — ${err.message}`);
      errors.push(`${name}: ${err.message}`);
    }
  }

  return res.status(503).json({
    error: 'All AI providers failed. Please try again shortly.',
    details: errors,
  });
});

// ── Health check ──────────────────────────────────────────────────────────────

app.get('/api/health', (_req, res) => {
  const configured = PROVIDERS
    .map(p => p.name)
    .filter((_, i) => {
      const keys = [process.env.ANTHROPIC_API_KEY, process.env.OPENAI_API_KEY, process.env.GEMINI_API_KEY];
      return !!keys[i];
    });
  res.json({ status: 'ok', providers: configured });
});

// ── Start ─────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  log('SERVER', `EimemesChat running → http://localhost:${PORT}`);
  const configured = PROVIDERS.filter((_, i) => {
    const keys = [process.env.ANTHROPIC_API_KEY, process.env.OPENAI_API_KEY, process.env.GEMINI_API_KEY];
    return !!keys[i];
  }).map(p => p.name);
  log('SERVER', configured.length ? `Providers: ${configured.join(' → ')}` : '⚠ No AI providers configured — check .env');
});

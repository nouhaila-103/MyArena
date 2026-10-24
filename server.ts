import express from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '25mb' }));

// Initialize GoogleGenAI server-side with telemetry header
const getGeminiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
};

// Word overlap calculation for code guard (threshold 0.60)
function computeWordOverlap(textA: string, textB: string): number {
  const cleanA = (textA || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2);
  const cleanB = (textB || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2);

  if (cleanA.length === 0 || cleanB.length === 0) return 0;
  const setA = new Set(cleanA);
  const setB = new Set(cleanB);
  let intersection = 0;
  for (const w of setA) {
    if (setB.has(w)) intersection++;
  }
  return intersection / Math.min(setA.size, setB.size);
}

function isValidLanguageScript(text: string, targetLang: 'French' | 'Arabic' | 'English'): boolean {
  if (!text || typeof text !== 'string') return false;
  const hasArabicScript = /[\u0600-\u06FF]/.test(text);
  if (targetLang === 'Arabic') {
    return hasArabicScript;
  }
  // For French or English, cannot be Arabic script
  if (hasArabicScript) return false;
  return true;
}

// Minimal local fallbacks in case API fails
function getLocalFallback(incidentType: string, seat: string, lockedLang: 'French' | 'Arabic' | 'English') {
  if (lockedLang === 'Arabic') {
    return {
      reply: "أنا معك، فرقة الإسعاف والمنظمون في طريقهم لمقعدك الآن. ابق بجانب الشخص واهدأ، واش كيتنفس؟",
      facts_update: { status: "help_en_route" },
      severity: "high",
      escalate: false,
      quick_replies: ["كيتنفس مزيان", "ما كيتنفسش", "المنظم وصل"],
      dashboard_summary: "Local safety fallback: help dispatched to Virage Nord seat; monitoring breathing.",
    };
  }
  if (lockedLang === 'English') {
    return {
      reply: "I am with you, safety stewards and responders are heading straight to your row. Stay beside the person, are they breathing?",
      facts_update: { status: "help_en_route" },
      severity: "high",
      escalate: false,
      quick_replies: ["Breathing normally", "Not breathing", "Steward is here"],
      dashboard_summary: "Local safety fallback: responders dispatched to seat; instructed to check breathing.",
    };
  }
  return {
    reply: "Je suis avec vous, les secours et les stadiers arrivent immédiatement à votre rangée. Restez près de la personne, respire-t-elle normalement ?",
    facts_update: { status: "help_en_route" },
    severity: "high",
    escalate: false,
    quick_replies: ["Elle respire", "Ne respire pas", "Un stadier arrive"],
    dashboard_summary: "Secours dépêchés à la rangée du supporter ; vérification de la respiration en cours.",
  };
}

// 1. SUPPORTER AI CHAT (REAL CHATBOT, NO SCRIPTED NODES)
app.post('/api/assistant-chat', async (req, res) => {
  const {
    incidentType = 'medical_emergency',
    seat = 'Gate B • Block 104 • Row 12 • Seat 44 • Virage Nord',
    lockedLanguage = 'French',
    history = [], // alternating user/model turns: [{ role: 'user'|'model', text: string }]
    supporterAnswer = '',
    isFirstMessage = false,
    state = {
      incident_type: 'medical_emergency',
      ticket_seat: 'Gate B • Block 104 • Row 12 • Seat 44 • Virage Nord',
      facts_known: {},
      actions_already_given: [],
    },
  } = req.body;

  // Normalize locked language strictly to French, Arabic, or English
  let lockedLang: 'French' | 'Arabic' | 'English' = 'French';
  const langLower = (lockedLanguage || '').toLowerCase().trim();
  if (langLower.startsWith('ar') || langLower.includes('عرب') || langLower.includes('darija')) {
    lockedLang = 'Arabic';
  } else if (langLower.startsWith('en') || langLower.includes('anglais')) {
    lockedLang = 'English';
  } else {
    lockedLang = 'French';
  }

  const ai = getGeminiClient();
  if (!ai) {
    const fallback = getLocalFallback(incidentType, seat, lockedLang);
    return res.json(fallback);
  }

  const systemPrompt = `You are the MyArena safety assistant, an AI inside a football stadium app, talking to one frightened person at ${seat}. Help is already dispatched. You are a calm, experienced steward having a real conversation: you listen first.
Rules:
1. Answer exactly what the person just said or asked before anything else. If they ask a question (what do I do, how long, who are you, is it serious, I don't understand), answer it directly. If they report something new (he is moving, she is breathing, people are running), react to it specifically.
2. Maximum two short spoken-style sentences. No lists, no formulas. Use reassurance like 'I'm with you' only once, in the first message.
3. Never repeat a sentence, instruction or question that already appears in the conversation. Every turn adds something new: a new action, a precise update, or a question about a fact not yet known. If nothing new is needed, describe what is happening now (the team's progress, what the steward will do) in different words.
4. Never ask for a fact already in the state. 'I don't know' (in any language) is a valid answer: give the safest default action and move forward.
5. Safety advice must come only from ALLOWED_ACTIONS, rephrased in your own words. Never diagnose, never promise outcomes.
6. If the person says someone is not breathing or unconscious, set escalate true and severity critical, and say responders were alerted with top priority.
7. Reply ONLY in ${lockedLang}. In Arabic, understand Darija and answer in simple Arabic with light Darija. In French, natural spoken French. In English, plain spoken English.
8. Match their energy: short panicked messages get very short answers.

ALLOWED_ACTIONS:
- Medical: Keep the person still and warm without moving them unless direct danger. Loosen tight collar or clothing around the neck. Check breathing and reaction; shout to nearest steward in yellow vest. If bleeding, press firmly on the wound with clean cloth without releasing. Do not give food or drink and do not force them to stand. If chest pain or breathing difficulty, keep them seated or semi-reclined.
- Fire / Smoke: Stay low under the smoke and cover mouth and nose with a garment. Head calmly toward lighted green emergency exit signs. Never touch flares or hot objects. Follow steward hand signals.
- Crowd crush / Surge: Cross arms over chest to protect breathing space. Move diagonally with crowd flow; do not fight against the crowd and never bend down for dropped items. Look for relief gates opened by stewards.
- Security / Brawl / Physical threat: Move calmly back several rows away from the altercation. Do not film or engage. Stay near stewards or family safety zones. Keep stairways clear for security teams.

Good exchange examples:
Example 1 (Medical - French):
User: "Elle s'est évanouie par terre !"
Model: {"reply": "Je suis avec vous, l'équipe médicale SAMU arrive à votre rangée. Ne la relevez pas et desserrez son col, est-ce qu'elle respire ?", "facts_update": {"victim_status": "fainted_on_ground"}, "severity": "high", "escalate": false, "quick_replies": ["Elle respire", "Ne respire pas", "Je ne sais pas"], "dashboard_summary": "Victime évanouie au sol ; col desserré, contrôle de la respiration en cours."}

Example 2 (Medical - Arabic/Darija):
User: "واحد طاح سخفان حدايا وما كيتنفسش"
Model: {"reply": "أنا معك، فرقة الإسعاف متوجهة دابا لصفكم في الفيراج. حل ليه الكول ديال القميص وخلي الناس يبعدو، واش كيتنفس؟", "facts_update": {"victim_status": "unconscious_unresponsive"}, "severity": "critical", "escalate": true, "quick_replies": ["كيتنفس", "ما كيتنفسش", "المنظم وصل"], "dashboard_summary": "شخص فاقد الوعي في المدرج الشمالي، تم توجيه الإسعاف بأولوية قصوى."}

Example 3 (Medical - English):
User: "My friend collapsed and is pale!"
Model: {"reply": "I am with you, medics are heading directly to your row. Keep them lying flat and do not force them to sit, are they breathing steadily?", "facts_update": {"victim_status": "collapsed_pale"}, "severity": "high", "escalate": false, "quick_replies": ["Breathing normally", "Struggling to breathe", "I don't know"], "dashboard_summary": "Supporter reported collapsed friend; instructed to keep them lying flat."}

Example 4 ('What do I do now?' / 'Que faire maintenant ?'):
User: "Que faire maintenant ?"
Model: {"reply": "Restez calme à ses côtés sans lui donner à boire. Le stadier avec la trousse de secours est à quelques mètres dans l'escalier.", "facts_update": {"action": "waiting_beside_victim"}, "severity": "high", "escalate": false, "quick_replies": ["Le stadier est là", "Toujours inconscient", "Il bouge"], "dashboard_summary": "Conseil d'attendre calmement sans donner d'eau ; stadier en approche."}

Example 5 ('I didn't understand' / 'Je n'ai pas compris'):
User: "Je n'ai pas compris"
Model: {"reply": "Ne touchez pas à la personne. Levez simplement le bras et criez vers le stadier en gilet jaune le plus proche.", "facts_update": {"action": "call_steward_in_yellow_vest"}, "severity": "high", "escalate": false, "quick_replies": ["J'ai appelé le stadier", "Il arrive", "Je ne vois personne"], "dashboard_summary": "Instruction simplifiée : lever le bras et appeler le stadier en gilet jaune."}

Example 6 ('I don't know' / 'ما كنعرفش'):
User: "ما كنعرفش واش كيتنفس"
Model: {"reply": "نادي دابا بصوت عالي على أقرب ستاديي حداكم. حط يدك بهدوء على صدرو واش كتشوفو كيطلع ويهبط؟", "facts_update": {"breathing_check": "uncertain"}, "severity": "high", "escalate": false, "quick_replies": ["الصدر كيتحرك", "ما كيتحركش", "المنظم وصل"], "dashboard_summary": "عدم التأكد من التنفس؛ توجيه بنداء المنظم وفحص حركة الصدر."}`;

  // Assemble FULL conversation history as alternating turns
  const contentsPayload: any[] = [];

  // Preamble state turn
  const currentStateJson = {
    incident_type: incidentType,
    ticket_seat: seat,
    locked_language: lockedLang,
    facts_known: state?.facts_known || {},
    actions_already_given: state?.actions_already_given || [],
  };

  // Convert incoming history to proper alternating turns
  if (history && Array.isArray(history) && history.length > 0) {
    for (const msg of history) {
      if (msg.role === 'user' || msg.sender === 'supporter') {
        contentsPayload.push({
          role: 'user',
          parts: [{ text: msg.text || '' }],
        });
      } else if (msg.role === 'model' || msg.sender === 'ai') {
        contentsPayload.push({
          role: 'model',
          parts: [{ text: msg.text || '' }],
        });
      }
    }
  }

  // Add the latest user message
  const userText = isFirstMessage
    ? `[EMERGENCY TRIGGER: ${incidentType.toUpperCase()} at ${seat}] Supporter tapped emergency button. Locked Language: ${lockedLang}. Current state: ${JSON.stringify(currentStateJson)}`
    : supporterAnswer || 'Help';

  contentsPayload.push({
    role: 'user',
    parts: [{ text: userText }],
  });

  // Extract previous assistant replies for word overlap check
  const previousAssistantReplies: string[] = [];
  if (history && Array.isArray(history)) {
    for (const msg of history) {
      if (msg.role === 'model' || msg.sender === 'ai') {
        if (msg.text) previousAssistantReplies.push(msg.text);
      }
    }
  }

  const callModel = async (retryPrompt?: string) => {
    const payload = retryPrompt
      ? [
          ...contentsPayload,
          {
            role: 'user',
            parts: [{ text: retryPrompt }],
          },
        ]
      : contentsPayload;

    return await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: payload,
      config: {
        systemInstruction: systemPrompt,
        temperature: 0.8,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            reply: { type: Type.STRING },
            facts_update: { type: Type.OBJECT },
            severity: {
              type: Type.STRING,
              enum: ['low', 'medium', 'high', 'critical'],
            },
            escalate: { type: Type.BOOLEAN },
            quick_replies: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            dashboard_summary: { type: Type.STRING },
          },
          required: ['reply', 'severity', 'escalate', 'quick_replies', 'dashboard_summary'],
        },
      },
    });
  };

  try {
    let response = await callModel();
    let parsed: any = null;

    if (response?.text) {
      try {
        parsed = JSON.parse(response.text);
      } catch (e) {
        console.error('Failed to parse model JSON:', e);
      }
    }

    // Code guard: check word overlap and script validity
    let shouldRegenerate = false;
    if (!parsed || !parsed.reply) {
      shouldRegenerate = true;
    } else {
      const scriptValid = isValidLanguageScript(parsed.reply, lockedLang);
      const overlap = previousAssistantReplies.length > 0
        ? Math.max(...previousAssistantReplies.map((p) => computeWordOverlap(parsed.reply, p)))
        : 0;

      if (!scriptValid || overlap > 0.6) {
        console.warn(`[MyArena Guard] Regenerating: scriptValid=${scriptValid}, maxOverlap=${overlap}`);
        shouldRegenerate = true;
      }
    }

    // Regenerate once if rejected
    if (shouldRegenerate) {
      const retryMsg = `Say something new and different, in ${lockedLang}. Follow all rules strictly.`;
      const retryResponse = await callModel(retryMsg);
      if (retryResponse?.text) {
        try {
          const retryParsed = JSON.parse(retryResponse.text);
          if (retryParsed && retryParsed.reply && isValidLanguageScript(retryParsed.reply, lockedLang)) {
            parsed = retryParsed;
          }
        } catch (_) {}
      }
    }

    if (parsed && parsed.reply) {
      return res.json({
        reply: parsed.reply,
        facts_update: parsed.facts_update || {},
        severity: parsed.severity || 'high',
        escalate: Boolean(parsed.escalate),
        quick_replies: Array.isArray(parsed.quick_replies) ? parsed.quick_replies.slice(0, 4) : [],
        dashboard_summary: parsed.dashboard_summary || `Supporter reported ${incidentType} at ${seat}`,
      });
    }

    // Fallback if model output was empty or invalid
    return res.json(getLocalFallback(incidentType, seat, lockedLang));
  } catch (err: any) {
    console.error('Error in /api/assistant-chat:', err?.message);
    return res.json(getLocalFallback(incidentType, seat, lockedLang));
  }
});

// 2. GEMINI TTS API
app.post('/api/tts', async (req, res) => {
  const { text, language = 'French' } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }

  const ai = getGeminiClient();
  if (!ai) {
    return res.status(503).json({ error: 'Gemini client unavailable' });
  }

  // Voice selection based on language
  const voiceName = 'Kore'; // Zephyr, Kore, Puck are high quality neutral voices

  try {
    // Primary: gemini-2.5-flash-preview-tts or gemini-3.8-flash-lite-tts
    let response: any = null;
    try {
      response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-preview-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: text,
                speechMetadata: {
                  style: 'Calm, clear, and reassuring football stadium safety steward',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName },
            },
          },
        },
      });
    } catch (ttsErr: any) {
      console.warn('gemini-2.5-flash-preview-tts failed, trying gemini-3.8-flash-lite-tts:', ttsErr?.message);
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-lite-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: text,
                speechMetadata: {
                  style: 'Calm, clear, and reassuring football stadium safety steward',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName },
            },
          },
        },
      });
    }

    const base64Audio = response?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      return res.json({
        audioBase64: base64Audio,
        mimeType: 'audio/wav',
      });
    }

    return res.status(502).json({ error: 'No audio returned from TTS model' });
  } catch (err: any) {
    console.error('Gemini TTS error:', err?.message);
    return res.status(500).json({ error: 'TTS generation failed', details: err?.message });
  }
});

// 3. GEMINI SPEECH INPUT TRANSCRIBE (INCLUDING DARIJA)
app.post('/api/transcribe-audio', async (req, res) => {
  const { audioBase64, mimeType = 'audio/webm' } = req.body;
  if (!audioBase64) {
    return res.status(400).json({ error: 'audioBase64 is required' });
  }

  const ai = getGeminiClient();
  if (!ai) {
    return res.status(503).json({ error: 'Gemini client unavailable' });
  }

  const cleanBase64 = audioBase64.replace(/^data:audio\/\w+;base64,/, '');

  try {
    const audioPart = {
      inlineData: {
        mimeType: mimeType || 'audio/webm',
        data: cleanBase64,
      },
    };

    const textPart = {
      text: `You are the audio speech transcriber for the MyArena football stadium safety assistant.
The speaker is an attendee at a match in Morocco or international venue.
They may speak Moroccan Darija (transcribe in standard Moroccan Darija Arabic letters or Latin Arabizi, exactly as spoken), Standard Arabic, French, or English.
Rules:
1. Transcribe the words accurately and verbatim.
2. Return ONLY the transcribed text. Do NOT include quotation marks, markdown, explanations, or metadata.`,
    };

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: { parts: [audioPart, textPart] },
    });

    const transcript = (response.text || '').trim();
    return res.json({ transcript });
  } catch (err: any) {
    console.error('Audio transcribe error:', err?.message);
    return res.status(500).json({ error: 'Transcription failed', details: err?.message });
  }
});

// 4. ANALYZE REPORT API
app.post('/api/analyze-report', async (req, res) => {
  const { text, zone, imageBase64 } = req.body;
  if (!text && !imageBase64) {
    return res.status(400).json({ error: 'Text or image is required' });
  }

  const ai = getGeminiClient();
  if (!ai) {
    return res.json({
      language: 'French',
      incident_type: 'medical_emergency',
      severity: 'high',
      zone: zone || 'Virage Nord',
      english_summary: `Supporter report in ${zone || 'Virage Nord'}: "${(text || '').slice(0, 80)}"`,
      recommended_action: 'Dispatch zone stewards to inspect',
      confidence: 0.85,
      source: 'fallback_engine',
    });
  }

  try {
    const promptText = `You are the AI Safety Engine for "MyArena", a premier football stadium security system.
Analyze the following incident report submitted by a football match supporter.
Supporters may write or speak in Moroccan Darija (Arabic letters or Latin script), French, Standard Arabic, or English.
Supporter seat / zone: "${zone || 'Gate B • Block 104 • Row 12 • Seat 44 • Virage Nord'}".
Supporter report text: "${text || '(Visual report from photo)'}"

Analyze the input:
1. Identify the input language (e.g. "Darija (Latin script)", "Darija (Arabic script)", "French", "Arabic", "English").
2. Determine incident type: "fight", "medical_emergency", "fire_smoke", "overcrowding", "theft", "other".
3. Evaluate severity: "low", "medium", "high", "critical".
4. Identify zone strictly from: "North Stand", "South Stand", "East Stand", "West Stand", "Gate A", "Gate B", "VIP", "Parking".
5. Provide a crisp 1-sentence English summary.
6. Provide an operational recommended security action.
7. Confidence score between 0.0 and 1.0.`;

    const contents: any[] = [];
    if (imageBase64) {
      const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');
      contents.push({
        inlineData: {
          mimeType: 'image/jpeg',
          data: cleanBase64,
        },
      });
    }
    contents.push({ text: promptText });

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: { parts: contents },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            language: { type: Type.STRING },
            incident_type: {
              type: Type.STRING,
              enum: ['fight', 'medical_emergency', 'fire_smoke', 'overcrowding', 'theft', 'other'],
            },
            severity: {
              type: Type.STRING,
              enum: ['low', 'medium', 'high', 'critical'],
            },
            zone: {
              type: Type.STRING,
              enum: ['North Stand', 'South Stand', 'East Stand', 'West Stand', 'Gate A', 'Gate B', 'VIP', 'Parking'],
            },
            english_summary: { type: Type.STRING },
            recommended_action: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
          },
          required: ['language', 'incident_type', 'severity', 'zone', 'english_summary', 'recommended_action', 'confidence'],
        },
      },
    });

    if (response.text) {
      return res.json({ ...JSON.parse(response.text), source: 'gemini' });
    }
    throw new Error('Empty response');
  } catch (err: any) {
    return res.json({
      language: 'French',
      incident_type: 'medical_emergency',
      severity: 'high',
      zone: zone || 'Virage Nord',
      english_summary: `Supporter report in ${zone || 'Virage Nord'}`,
      recommended_action: 'Dispatch stewards to verify seat',
      confidence: 0.8,
      source: 'fallback_engine',
    });
  }
});

// 5. DEDUPLICATION / MERGE API
app.post('/api/deduplicate', async (req, res) => {
  const { newReport, openIncidents } = req.body;
  if (!newReport) {
    return res.status(400).json({ error: 'newReport is required' });
  }

  if (!openIncidents || openIncidents.length === 0) {
    return res.json({
      matchedIncidentId: null,
      shouldMerge: false,
      reason: 'No open incidents currently in stadium queue.',
    });
  }

  const ai = getGeminiClient();
  if (!ai) {
    const match = openIncidents.find(
      (inc: any) => inc.zone === newReport.zone && inc.incident_type === newReport.incident_type
    );
    if (match) {
      return res.json({
        matchedIncidentId: match.id,
        shouldMerge: true,
        why_merged: `Corroborates active ${match.incident_type} in ${match.zone} reported within matching timeframe.`,
      });
    }
    return res.json({
      matchedIncidentId: null,
      shouldMerge: false,
      why_merged: 'Treated as independent incident.',
    });
  }

  try {
    const candidates = openIncidents.map((inc: any) => ({
      id: inc.id,
      zone: inc.zone,
      type: inc.incident_type,
      summary: inc.english_summary,
      reportCount: inc.reportCount,
      lastReportTime: inc.lastReportTime,
    }));

    const dedupePrompt = `You are the stadium incident deduplication engine for MyArena.
A new incident report has arrived:
- Zone: ${newReport.zone}
- Incident Type: ${newReport.incident_type}
- Original Supporter Text: "${newReport.raw_text || newReport.text}"

Active incidents:
${JSON.stringify(candidates, null, 2)}

Rules:
If they describe the SAME event, return shouldMerge: true, matchedIncidentId: <id>, and a concise why_merged explanation. Otherwise shouldMerge: false, matchedIncidentId: null.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: dedupePrompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            shouldMerge: { type: Type.BOOLEAN },
            matchedIncidentId: { type: Type.STRING },
            why_merged: { type: Type.STRING },
          },
          required: ['shouldMerge', 'why_merged'],
        },
      },
    });

    if (response.text) {
      return res.json(JSON.parse(response.text));
    }
    throw new Error('Empty dedupe response');
  } catch (err: any) {
    const match = openIncidents.find(
      (inc: any) => inc.zone === newReport.zone && inc.incident_type === newReport.incident_type
    );
    if (match) {
      return res.json({
        matchedIncidentId: match.id,
        shouldMerge: true,
        why_merged: `Corroborates active ${match.incident_type} in ${match.zone}.`,
      });
    }
    return res.json({
      matchedIncidentId: null,
      shouldMerge: false,
      why_merged: 'Treated as independent new incident.',
    });
  }
});

// 6. GENERATE FORMAL INCIDENT REPORT IN FRENCH
app.post('/api/generate-security-report', async (req, res) => {
  const { incidents, stadiumInfo } = req.body;
  const ai = getGeminiClient();

  const fallbackReport = `================================================================================
                    RAPPORT OFFICIEL D'INCIDENT DE SÉCURITÉ
                           MYARENA STADIUM SAFETY OPS
================================================================================
Date & Heure : ${new Date().toLocaleString('fr-FR')}
Événement    : Match officiel - Grand Stade
Statut Global: NIVEAU DE VIGILANCE RENFORCÉ

I. SYNTHÈSE GLOBALE
--------------------------------------------------------------------------------
Nombre total d'incidents enregistrés : ${incidents?.length || 0}
Siège du supporter référencé       : Gate B • Block 104 • Row 12 • Seat 44 • Virage Nord

II. DÉTAILS DES INCIDENTS PAR SECTEUR
--------------------------------------------------------------------------------
${(incidents || [])
  .map(
    (inc: any, idx: number) => `
[INCIDENT #${idx + 1}] - ZONE : ${inc.exactSeat || inc.zone}
- Typologie         : ${inc.incident_type?.toUpperCase()}
- Gravité           : ${inc.severity?.toUpperCase()}
- Signalements liés : ${inc.reportCount || 1} rapport(s)
- Résumé opérationnel : ${inc.dashboard_summary || inc.english_summary || 'Aucun résumé'}
- Mesures requises  : ${inc.recommended_action || 'Contrôle par les stadiers'}
- Justification     : ${inc.why_merged || 'Premier signalement'}
`
  )
  .join('\n')}

III. RECOMMANDATIONS DU COMMANDEMENT
--------------------------------------------------------------------------------
1. Maintenir le dispositif de sécurité actif aux abords de la Porte B et du Virage Nord.
2. Garantir les couloirs d'évacuation médicale vers les issues principales.
3. Rapport transmis à la préfecture et au délégué du match.
================================================================================`;

  if (!ai) {
    return res.json({ report: fallbackReport, source: 'fallback_engine' });
  }

  try {
    const prompt = `Génère un rapport officiel d'incident de sécurité de stade, rédigé en Français professionnel, conforme aux exigences de la préfecture de police, de la commission de sécurité de la ligue et du délégué du match.
Données des incidents en temps réel :
${JSON.stringify(incidents, null, 2)}
Stade : ${stadiumInfo?.name || 'Grand Stade - MyArena'}
Siège du supporter enregistré : Gate B • Block 104 • Row 12 • Seat 44 • Virage Nord

Le rapport doit être clair, formel, structuré :
1. EN-TÊTE OFFICIEL (Date, heure, statut du stade, niveau d'alerte global)
2. RÉSUMÉ EXÉCUTIF (Alertes reçues des supporters en Darija/Français/Arabe, incidents traités et fusionnés par l'IA)
3. FICHE DÉTAILLÉE DES ÉVÉNEMENTS (Secteur exact du siège, type d'incident, degré d'urgence, action entreprise par les secours)
4. PLAN D'ACTION IMMÉDIAT ET DISPOSITIF DE CLÔTURE.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        systemInstruction: "Tu es le responsable en chef de la sécurité d'un grand stade de football international.",
      },
    });

    if (response.text) {
      return res.json({ report: response.text, source: 'gemini' });
    }
    return res.json({ report: fallbackReport, source: 'fallback_engine' });
  } catch (err: any) {
    console.error('Failed to generate formal report with Gemini:', err?.message);
    return res.json({ report: fallbackReport, source: 'fallback_engine' });
  }
});

// Setup Vite middleware in dev or static files in production
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[MyArena Server] Running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Error starting server:', err);
});

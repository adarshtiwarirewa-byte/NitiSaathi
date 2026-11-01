import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const samples = [
  "Am I eligible?",
  "What benefits can I get?",
  "What documents do I need?",
  "What should I do next?",
];

const LANGUAGE_OPTIONS = [
  { value: "Hindi", label: "हिन्दी (Hindi)" },
  { value: "English", label: "English (India)" },
  { value: "Bengali", label: "বাংলা (Bengali)" },
  { value: "Gujarati", label: "ગુજરાતી (Gujarati)" },
  { value: "Kannada", label: "ಕನ್ನಡ (Kannada)" },
  { value: "Marathi", label: "मराठी (Marathi)" },
  { value: "Punjabi", label: "ਪੰਜਾਬੀ (Punjabi)" },
  { value: "Tamil", label: "தமிழ் (Tamil)" },
  { value: "Telugu", label: "తెలుగు (Telugu)" },
] as const;

type LanguageName = (typeof LANGUAGE_OPTIONS)[number]["value"];
type ApiStatus = "checking" | "online" | "offline";
type SpeechStatus = "idle" | "generating" | "playing";

type SpeechConfig = {
  language_code: string;
  speaker: string;
};

// Hindi and English retain their already-tested Bulbul v4 voices.
// Regional languages use Sarvam's documented language-specific Bulbul v3 voices.
const SPEECH_CONFIG: Record<LanguageName, SpeechConfig> = {
  Hindi: {
    language_code: "hi-IN",
    speaker: "aparna_hi_customer",
  },
  English: {
    language_code: "en-IN",
    speaker: "simran_en_customer",
  },
  Bengali: {
    language_code: "bn-IN",
    speaker: "roopa",
  },
  Gujarati: {
    language_code: "gu-IN",
    speaker: "priya",
  },
  Kannada: {
    language_code: "kn-IN",
    speaker: "ishita",
  },
  Marathi: {
    language_code: "mr-IN",
    speaker: "priya",
  },
  Punjabi: {
    language_code: "pa-IN",
    speaker: "roopa",
  },
  Tamil: {
    language_code: "ta-IN",
    speaker: "ishita",
  },
  Telugu: {
    language_code: "te-IN",
    speaker: "priya",
  },
};

const samplesByLanguage: Record<LanguageName, string[]> = {
  Hindi: [
    "क्या मैं इस योजना के लिए पात्र हूँ?",
    "मुझे कौन से लाभ मिल सकते हैं?",
    "कौन से दस्तावेज़ चाहिए?",
    "मुझे आगे क्या करना चाहिए?",
  ],
  English: samples,
  Bengali: [
    "আমি কি এই প্রকল্পের জন্য যোগ্য?",
    "আমি কী কী সুবিধা পেতে পারি?",
    "কোন নথিগুলি লাগবে?",
    "আমার পরবর্তী পদক্ষেপ কী?",
  ],
  Gujarati: [
    "શું હું આ યોજના માટે પાત્ર છું?",
    "મને કયા લાભો મળી શકે?",
    "કયા દસ્તાવેજોની જરૂર છે?",
    "મારે આગળ શું કરવું જોઈએ?",
  ],
  Kannada: [
    "ಈ ಯೋಜನೆಗೆ ನಾನು ಅರ್ಹನೇ?",
    "ನನಗೆ ಯಾವ ಪ್ರಯೋಜನಗಳು ಸಿಗುತ್ತವೆ?",
    "ಯಾವ ದಾಖಲೆಗಳು ಬೇಕು?",
    "ನಾನು ಮುಂದೆ ಏನು ಮಾಡಬೇಕು?",
  ],
  Marathi: [
    "मी या योजनेसाठी पात्र आहे का?",
    "मला कोणते लाभ मिळू शकतात?",
    "कोणती कागदपत्रे आवश्यक आहेत?",
    "मी पुढे काय करावे?",
  ],
  Punjabi: [
    "ਕੀ ਮੈਂ ਇਸ ਯੋਜਨਾ ਲਈ ਯੋਗ ਹਾਂ?",
    "ਮੈਨੂੰ ਕਿਹੜੇ ਲਾਭ ਮਿਲ ਸਕਦੇ ਹਨ?",
    "ਕਿਹੜੇ ਦਸਤਾਵੇਜ਼ ਚਾਹੀਦੇ ਹਨ?",
    "ਮੈਨੂੰ ਅੱਗੇ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?",
  ],
  Tamil: [
    "இந்தத் திட்டத்திற்கு நான் தகுதியானவரா?",
    "எனக்கு என்னென்ன நன்மைகள் கிடைக்கும்?",
    "என்ன ஆவணங்கள் தேவை?",
    "அடுத்து நான் என்ன செய்ய வேண்டும்?",
  ],
  Telugu: [
    "ఈ పథకానికి నేను అర్హుడినా?",
    "నాకు ఏ ప్రయోజనాలు లభిస్తాయి?",
    "ఏ పత్రాలు అవసరం?",
    "నేను తర్వాత ఏమి చేయాలి?",
  ],
};

type DocumentInfo = {
  document_id: string;
  filename: string;
  page_count: number;
};

type Citation = {
  source_number: number;
  document_id?: string;
  filename: string;
  page_number?: number;
  url?: string;
  domain?: string;
  source_type?: string;
  retrieved_at?: string;
};

type Point = {
  text: string;
  source_numbers: number[];
  evidence_basis: string;
};

type Section = {
  status: string;
  points: Point[];
};

type Explanation = {
  source_mode?: "document" | "web_research";
  question: string;
  answer_language: string;
  sources: Citation[];
  information_not_found: string[];
  policy_overview: Section;
  eligibility: Section;
  rights_and_benefits: Section;
  obligations: Section;
  application_process: Section;
  deadlines_and_conditions: Section;
  real_world_impact: Section;
};

type PageData = {
  filename: string;
  page_number: number;
  text: string;
};

const sections: [keyof Explanation, string][] = [
  ["policy_overview", "Policy at a glance"],
  ["eligibility", "Who may qualify"],
  ["rights_and_benefits", "Rights and benefits"],
  ["obligations", "What you need to do"],
  ["application_process", "How to apply"],
  ["deadlines_and_conditions", "Dates and conditions"],
  ["real_world_impact", "What this could mean for you"],
];

function splitSpeechText(text: string, maxLength = 2200): string[] {
  const sentences = text.match(/[^.!?।]+[.!?।]*/gu) || [text];
  const chunks: string[] = [];
  let current = "";

  for (const rawSentence of sentences) {
    const sentence = rawSentence.trim();
    if (!sentence) continue;

    if (sentence.length > maxLength) {
      if (current) {
        chunks.push(current);
        current = "";
      }

      let remaining = sentence;
      while (remaining.length > maxLength) {
        let splitAt = remaining.lastIndexOf(" ", maxLength);
        if (splitAt < maxLength * 0.5) splitAt = maxLength;
        chunks.push(remaining.slice(0, splitAt).trim());
        remaining = remaining.slice(splitAt).trim();
      }
      current = remaining;
      continue;
    }

    const combined = current ? `${current} ${sentence}` : sentence;
    if (combined.length > maxLength) {
      if (current) chunks.push(current);
      current = sentence;
    } else {
      current = combined;
    }
  }

  if (current) chunks.push(current);
  return chunks;
}

function getApiErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const data = payload as Record<string, unknown>;
  const detail = data.detail;

  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg);
        }
        return String(item);
      })
      .join("; ");
  }
  if (typeof data.message === "string") return data.message;
  return fallback;
}

function App() {
  const [language, setLanguage] = useState<LanguageName>("Hindi");
  const [file, setFile] = useState<File | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [documentId, setDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [busy, setBusy] = useState(false);
  const [speechStatus, setSpeechStatus] = useState<SpeechStatus>("idle");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [answer, setAnswer] = useState<Explanation | null>(null);
  const [page, setPage] = useState<PageData | null>(null);
  const [pageDocumentId, setPageDocumentId] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speechAbortRef = useRef<AbortController | null>(null);

  const refreshDocuments = async (): Promise<DocumentInfo[]> => {
    const response = await fetch(`${API_URL}/documents`);
    if (!response.ok) throw new Error("Could not load the policy library.");
    const data = (await response.json()) as { documents: DocumentInfo[] };
    setDocuments(data.documents);
    return data.documents;
  };

  const stopSpeech = () => {
    speechAbortRef.current?.abort();
    speechAbortRef.current = null;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }

    setSpeechStatus("idle");
  };

  useEffect(() => {
    let mounted = true;

    async function initialize() {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) throw new Error("Backend unavailable");
        await refreshDocuments();

        if (mounted) {
          // No auto-selected PDF: an empty selection starts web research.
          setApiStatus("online");
        }
      } catch {
        if (mounted) setApiStatus("offline");
      }
    }

    void initialize();
    return () => {
      mounted = false;
      speechAbortRef.current?.abort();
      audioRef.current?.pause();
    };
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (!selected) return;

    if (!selected.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF document.");
      event.target.value = "";
      return;
    }

    if (selected.size > 20 * 1024 * 1024) {
      setError("This PDF is larger than the 20 MB upload limit.");
      event.target.value = "";
      return;
    }

    stopSpeech();
    setError("");
    setFile(selected);
    setAnswer(null);
    setPage(null);
    setNotice("");
  };

  const upload = async () => {
    if (!file) return;

    stopSpeech();
    setBusy(true);
    setError("");
    setNotice("Reading pages and preparing searchable passages…");

    try {
      const body = new FormData();
      body.append("file", file);
      body.append("document_language", "auto");

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body,
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          getApiErrorMessage(data, "The PDF could not be processed."),
        );
      }

      const doc = data.document as DocumentInfo;
      setFile(null);
      setNotice(`${doc.filename} is ready · ${doc.page_count} pages indexed`);
      await refreshDocuments();
      setDocumentId(doc.document_id);
      setAnswer(null);
      setPage(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed.");
      setNotice("");
    } finally {
      setBusy(false);
    }
  };

  const ask = async (prompt = question) => {
    if (!prompt.trim()) {
      setError("Write a question or choose a sample question first.");
      return;
    }
    stopSpeech();
    setQuestion(prompt);
    setBusy(true);
    setError("");
    setAnswer(null);
    setPage(null);

    try {
      const endpoint = documentId ? "/documents/ask" : "/research/ask";
      const payload = documentId
        ? { question: prompt, language, top_k: 4, document_id: documentId }
        : { question: prompt, language, top_k: 6 };
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          getApiErrorMessage(
            data,
            "The explanation could not be generated.",
          ),
        );
      }

      setAnswer(data.result as Explanation);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  };

  const openCitation = async (source: Citation) => {
    if (source.url) {
      window.open(source.url, "_blank", "noopener,noreferrer");
      return;
    }
    if (!source.document_id || !source.page_number) {
      setError("This source cannot be opened.");
      return;
    }
    try {
      const response = await fetch(
        `${API_URL}/documents/${source.document_id}/pages/${source.page_number}`,
      );
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          getApiErrorMessage(data, "Could not open the cited page."),
        );
      }

      setPage(data as PageData);
      setPageDocumentId(source.document_id);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not open source.",
      );
    }
  };

  const speak = async () => {
    if (speechStatus !== "idle") {
      stopSpeech();
      return;
    }
    if (!answer) return;

    const speechConfig = SPEECH_CONFIG[language];
    const answerText = sections
      .flatMap(([key]) => {
        const section = answer[key] as Section;
        return section?.points?.map((point) => point.text) || [];
      })
      .join(". ");

    const completeText = [
      answerText,
      ...(answer.information_not_found || []),
    ].filter(Boolean).join(". ");

    const chunks = splitSpeechText(completeText);
    if (!chunks.length) {
      setError("There is no explanation text available to read aloud.");
      return;
    }

    const controller = new AbortController();
    speechAbortRef.current = controller;
    setError("");

    try {
      for (const chunk of chunks) {
        if (controller.signal.aborted) break;
        setSpeechStatus("generating");

        const response = await fetch(`${API_URL}/documents/speak`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: chunk,
            language_code: speechConfig.language_code,
            speaker: speechConfig.speaker,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          throw new Error(
            getApiErrorMessage(
              payload,
              `Speech generation failed (${response.status}).`,
            ),
          );
        }

        const audioBlob = await response.blob();
        if (!audioBlob.size) throw new Error("Sarvam returned an empty audio file.");
        if (controller.signal.aborted) break;

        setSpeechStatus("playing");

        await new Promise<void>((resolve, reject) => {
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          audioRef.current = audio;
          let settled = false;

          const cleanup = () => {
            URL.revokeObjectURL(audioUrl);
            audio.removeEventListener("ended", onEnded);
            audio.removeEventListener("error", onError);
            controller.signal.removeEventListener("abort", onAbort);
            if (audioRef.current === audio) audioRef.current = null;
          };
          const finish = () => {
            if (settled) return;
            settled = true;
            cleanup();
            resolve();
          };
          const onEnded = () => finish();
          const onAbort = () => {
            audio.pause();
            audio.currentTime = 0;
            finish();
          };
          const onError = () => {
            if (settled) return;
            settled = true;
            cleanup();
            reject(new Error("The generated audio could not be played."));
          };

          audio.addEventListener("ended", onEnded);
          audio.addEventListener("error", onError);
          controller.signal.addEventListener("abort", onAbort, { once: true });
          audio.play().catch(onError);
        });
      }
    } catch (reason) {
      if (
        !(reason instanceof DOMException && reason.name === "AbortError") &&
        !controller.signal.aborted
      ) {
        setError(
          reason instanceof Error
            ? reason.message
            : "Could not generate speech. Please try again.",
        );
      }
    } finally {
      if (speechAbortRef.current === controller) {
        speechAbortRef.current = null;
        setSpeechStatus("idle");
      }
    }
  };

  const statusLabel =
    apiStatus === "checking"
      ? "Connecting"
      : apiStatus === "online"
        ? "System online"
        : "Backend offline";

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="#top" aria-label="NitiSaathi home">
          <span className="brand-mark">न</span>
          <span>
            <strong>NitiSaathi</strong>
            <small>POLICY CLARITY FOR EVERYONE</small>
          </span>
        </a>
        <nav className="nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="#trust">Our commitment</a>
        </nav>
        <div className="api-indicator">
          <span className={`status-dot ${apiStatus}`} />
          {statusLabel}
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              POLICIES, RIGHTS & SCHEMES — MADE HUMAN
            </div>
            <h1>
              Know what the policy means.
              <br />
              <span>Find your next step.</span>
            </h1>
            <p className="hero-description">
              Turn government notifications, welfare schemes and public rules
              into clear explanations in your language, with every key point
              linked to its source page.
            </p>
            <div className="hero-tags">
              <span>9 Indian languages</span>
              <span>Page-linked evidence</span>
              <span>AI voice narration</span>
            </div>
          </div>

          <aside className="trust-card">
            <div className="trust-icon">✓</div>
            <h3>Built around the source</h3>
            <p>
              We find relevant passages first, then explain what they say in
              everyday language.
            </p>
            <div className="trust-divider" />
            <div className="trust-point"><span>01</span>Evidence before answers</div>
            <div className="trust-point"><span>02</span>Explicit facts vs interpretation</div>
            <div className="trust-point"><span>03</span>Honest when information is missing</div>
          </aside>
        </section>

        <section className="workspace" id="how-it-works">
          <div className="section-heading">
            <div>
              <span className="section-eyebrow">YOUR POLICY DESK</span>
              <h2>Understand a policy.</h2>
              <p>Choose a PDF or ask without one. NitiSaathi will find the evidence.</p>
            </div>
            <span className="step-label">2 SIMPLE STEPS</span>
          </div>

          <div className="workspace-grid">
            <section className="panel document-panel">
              <div className="panel-heading">
                <span className="step-number">01</span>
                <div>
                  <h3>Choose a document</h3>
                  <p>Use a policy already here or upload a PDF.</p>
                </div>
              </div>

              {documents.length > 0 && (
                <>
                  <label className="field-label" htmlFor="document">YOUR POLICY LIBRARY</label>
                  <select
                    id="document"
                    className="language-select document-select"
                    value={documentId}
                    onChange={(event) => {
                      stopSpeech();
                      setDocumentId(event.target.value);
                      setAnswer(null);
                      setPage(null);
                      setError("");
                    }}
                  >
                    <option value="">Choose a policy</option>
                    {documents.map((doc) => (
                      <option key={doc.document_id} value={doc.document_id}>
                        {doc.filename} · {doc.page_count} pages
                      </option>
                    ))}
                  </select>
                  <div className="or-divider"><span>OR ADD A POLICY</span></div>
                </>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                hidden
                onChange={handleFileChange}
              />
              <button
                className={`upload-area ${file ? "has-file" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                type="button"
                disabled={busy}
              >
                <span className="upload-icon">{file ? "✓" : "↑"}</span>
                {file ? (
                  <>
                    <strong>{file.name}</strong>
                    <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                    <small>Click to choose another PDF</small>
                  </>
                ) : (
                  <>
                    <strong>Upload a policy document</strong>
                    <span>Choose a PDF from your device</span>
                    <small>PDF only · Up to 20 MB</small>
                  </>
                )}
              </button>

              {file && (
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy}
                  onClick={() => void upload()}
                >
                  {busy ? "Preparing document…" : "Add this document"}
                  <span>→</span>
                </button>
              )}
              <div className="privacy-note">
                <span>⌑</span>
                For this demo, use public sample documents. Avoid personal information.
              </div>
            </section>

            <section className="panel question-panel">
              <div className="panel-heading">
                <span className="step-number">02</span>
                <div>
                  <h3>Ask what matters to you</h3>
                  <p>Pick a language and start with a real-world question.</p>
                </div>
              </div>

              <label className="field-label" htmlFor="language">EXPLANATION LANGUAGE</label>
              <select
                id="language"
                className="language-select"
                value={language}
                onChange={(event) => {
                  stopSpeech();
                  setLanguage(event.target.value as LanguageName);
                  setAnswer(null);
                  setError("");
                }}
              >
                {LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <label className="field-label question-label" htmlFor="question">
                WHAT WOULD YOU LIKE TO KNOW?
              </label>
              <textarea
                id="question"
                className="question-input"
                placeholder={samplesByLanguage[language][0]}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                    void ask();
                  }
                }}
                rows={3}
              />

              <div className="sample-heading">NOT SURE WHERE TO START?</div>
              <div className="sample-questions">
                {samplesByLanguage[language].map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => setQuestion(sample)}
                    className="sample-chip"
                  >
                    {sample}<span>↗</span>
                  </button>
                ))}
              </div>

              <button
                className="ask-button"
                type="button"
                disabled={busy || apiStatus !== "online"}
                onClick={() => void ask()}
              >
                {busy
                  ? (documentId ? "Finding document evidence…" : "Researching government sources…")
                  : (documentId ? "Explain this policy" : "Research government sources")}
                <span>→</span>
              </button>
              <p className="coming-soon">
                {documentId
                  ? "Using your selected PDF · Shortcut: Ctrl + Enter"
                  : "No PDF selected — NitiSaathi will automatically research reliable web sources."}
              </p>
            </section>
          </div>

          {(error || notice) && (
            <div
              className={`feedback ${error ? "feedback-error" : "feedback-success"}`}
              role="status"
            >
              {error || notice}
              <button
                type="button"
                onClick={() => { setError(""); setNotice(""); }}
                aria-label="Dismiss"
              >
                ×
              </button>
            </div>
          )}

          {answer && (
            <section className="answer-card" aria-live="polite">
              <div className="answer-heading">
                <div>
                  <span className="section-eyebrow">
                    YOUR EXPLANATION · {answer.answer_language}
                  </span>
                  <h2>{answer.question}</h2>
                </div>
                <button
                  type="button"
                  className="listen-button"
                  onClick={() => void speak()}
                  disabled={busy}
                >
                  {speechStatus === "idle"
                    ? "▶ Read aloud"
                    : speechStatus === "generating"
                      ? "Preparing audio… · Stop"
                      : "Playing audio · Stop"}
                </button>
              </div>

              <div className="answer-grid">
                <div className="answer-sections">
                  {sections.map(([key, title]) => {
                    const section = answer[key] as Section;
                    if (!section || section.status !== "available" || !section.points.length) {
                      return null;
                    }
                    return (
                      <article className="answer-section" key={key}>
                        <h3>{title}</h3>
                        {section.points.map((point, index) => (
                          <div className="answer-point" key={`${key}-${index}`}>
                            <p>{point.text}</p>
                            <span className="basis-label">
                              {point.evidence_basis === "cautious_interpretation"
                                ? "Plain-language interpretation"
                                : answer.source_mode === "web_research"
                                  ? "Supported by web evidence"
                                  : "Stated in the document"}
                            </span>
                            <div className="point-citations">
                              {point.source_numbers.map((number) => {
                                const source = answer.sources.find(
                                  (item) => item.source_number === number,
                                );
                                return source ? (
                                  <button
                                    className="citation-chip"
                                    type="button"
                                    key={number}
                                    onClick={() => void openCitation(source)}
                                  >
                                    {source.page_number
                                      ? `Source ${number} · p. ${source.page_number} ↗`
                                      : `Source ${number} · web ↗`}
                                  </button>
                                ) : null;
                              })}
                            </div>
                          </div>
                        ))}
                      </article>
                    );
                  })}

                  {answer.information_not_found.length > 0 && (
                    <div className="missing-info">
                      <strong>{answer.source_mode === "web_research"
                        ? "What reliable sources did not establish"
                        : "What the document doesn’t tell us"}</strong>
                      <ul>
                        {answer.information_not_found.map((item, index) => (
                          <li key={index}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {!answer.sources.length && (
                    <div className="missing-info">
                      <strong>No supporting passage found</strong>
                      <p>
                        {answer.source_mode === "web_research"
                          ? "The search did not provide enough source-linked evidence to answer this question."
                          : "The selected document did not provide enough evidence to answer this question. Try asking about a topic that appears in it."}
                      </p>
                    </div>
                  )}
                </div>

                <aside className="source-list">
                  <h3>Evidence used</h3>
                  {answer.sources.length ? (
                    answer.sources.map((source) => (
                      <button
                        type="button"
                        className="source-card"
                        key={source.source_number}
                        onClick={() => void openCitation(source)}
                      >
                        <span>
                          Source {source.source_number}
                          <b>{source.page_number
                            ? `Page ${source.page_number}`
                            : source.source_type === "official_government"
                              ? "Official government"
                              : "Supporting article"}</b>
                        </span>
                        <strong>{source.filename}</strong>
                        <small>{source.domain || "Document source"} · Open source ↗</small>
                      </button>
                    ))
                  ) : (
                    <p className="source-empty">
                      Sources appear here when relevant passages are found.
                    </p>
                  )}
                </aside>
              </div>

              <p className="answer-disclaimer">
                {answer.source_mode === "web_research"
                  ? "This is a source-grounded web research explanation, not legal advice or an official eligibility decision. Confirm current requirements with the issuing department."
                  : "This is a plain-language explanation of the selected document, not legal advice or an official eligibility decision."}
              </p>
            </section>
          )}

          {page && (
            <section className="page-viewer" id="source-page">
              <div className="page-heading">
                <div>
                  <span className="section-eyebrow">ORIGINAL SOURCE</span>
                  <h2>{page.filename} · Page {page.page_number}</h2>
                </div>
                <div className="page-actions">
                  <a
                    href={`${API_URL}/documents/${pageDocumentId}/file#page=${page.page_number}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open PDF ↗
                  </a>
                  <button type="button" onClick={() => setPage(null)} aria-label="Close source">
                    ×
                  </button>
                </div>
              </div>
              <p className="page-text">
                {page.text || "No extractable text is available for this page."}
              </p>
            </section>
          )}
        </section>

        <section className="bottom-note" id="trust">
          <span>✳</span>
          <p>
            NitiSaathi explains what a source document says. It is not legal advice
            and does not make official government decisions. Always confirm
            time-sensitive details with the issuing authority.
          </p>
        </section>
      </main>

      <footer className="footer">
        <span>© 2026 NitiSaathi</span>
        <span>9 Indian languages · PDF and web research</span>
      </footer>
    </div>
  );
}

export default App;

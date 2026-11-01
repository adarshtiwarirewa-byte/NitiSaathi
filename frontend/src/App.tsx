import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const sampleQuestions = [
  "Am I eligible for this scheme?",
  "What benefits can I receive?",
  "Which documents do I need?",
  "What should I do next?",
];

type ApiStatus = "checking" | "online" | "offline";
type UploadStatus = "idle" | "uploading" | "uploaded" | "error";
type AskStatus = "idle" | "asking";

type UploadedDocument = {
  document_id: string;
  filename: string;
  page_count: number;
  indexed_chunks: number;
  warnings?: string[];
};

type AskAnswer = {
  answer: string;
  language: string;
  sources: Array<{ filename: string; page_number: number }>;
};

type ApiErrorPayload = {
  detail?: unknown;
  message?: unknown;
};

function getApiErrorMessage(
  payload: ApiErrorPayload | null,
  fallback: string,
) {
  const detail = payload?.detail;

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
  if (typeof payload?.message === "string") return payload.message;
  return fallback;
}

function App() {
  const [language, setLanguage] = useState("Hindi");
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadedDocument, setUploadedDocument] =
    useState<UploadedDocument | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [question, setQuestion] = useState("");
  const [askStatus, setAskStatus] = useState<AskStatus>("idle");
  const [askError, setAskError] = useState("");
  const [answer, setAnswer] = useState<AskAnswer | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check whether our FastAPI backend is running.
  useEffect(() => {
    let isMounted = true;

    fetch(`${API_URL}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend is unavailable");
        }
        return response.json();
      })
      .then(() => {
        if (isMounted) setApiStatus("online");
      })
      .catch(() => {
        if (isMounted) setApiStatus("offline");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const uploadDocument = async (selectedFile: File) => {
    setUploadStatus("uploading");
    setUploadError("");
    setUploadedDocument(null);
    setAnswer(null);
    setAskError("");
    setAskStatus("idle");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json().catch(() => null)) as
        | ({ success?: boolean; document?: UploadedDocument } & ApiErrorPayload)
        | null;

      if (!response.ok) {
        throw new Error(
          getApiErrorMessage(payload, `Upload failed (${response.status}).`),
        );
      }
      if (!payload?.success || !payload.document) {
        throw new Error("The backend did not return uploaded document details.");
      }

      setUploadedDocument(payload.document);
      setUploadStatus("uploaded");
    } catch (error) {
      setUploadStatus("error");
      setUploadError(
        error instanceof Error
          ? error.message
          : "The PDF could not be uploaded.",
      );
    }
  };

  // Validate and upload the selected PDF to the backend.
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const selectedFile = input.files?.[0];
    input.value = "";

    if (!selectedFile) return;

    setFile(selectedFile);
    setUploadStatus("idle");
    setUploadedDocument(null);
    setUploadError("");
    setAnswer(null);
    setAskError("");

    if (
      selectedFile.type !== "application/pdf" &&
      !selectedFile.name.toLowerCase().endsWith(".pdf")
    ) {
      setUploadStatus("error");
      setUploadError("Please select a PDF document.");
      return;
    }

    if (selectedFile.size > 20 * 1024 * 1024) {
      setUploadStatus("error");
      setUploadError("This PDF is larger than the 20 MB upload limit.");
      return;
    }

    void uploadDocument(selectedFile);
  };

  const handleAsk = async () => {
    const trimmedQuestion = question.trim();
    if (!uploadedDocument || !trimmedQuestion) return;

    setAskStatus("asking");
    setAskError("");
    setAnswer(null);

    try {
      const response = await fetch(`${API_URL}/documents/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmedQuestion,
          language,
          top_k: 3,
        }),
      });
      const payload = (await response.json().catch(() => null)) as
        | ({ success?: boolean; result?: AskAnswer } & ApiErrorPayload)
        | null;

      if (!response.ok) {
        throw new Error(
          getApiErrorMessage(payload, `Could not get an answer (${response.status}).`),
        );
      }
      if (!payload?.success || !payload.result?.answer) {
        throw new Error("The backend did not return an answer.");
      }

      setAnswer(payload.result);
    } catch (error) {
      setAskError(
        error instanceof Error
          ? error.message
          : "Could not get an answer. Please try again.",
      );
    } finally {
      setAskStatus("idle");
    }
  };

  const formatFileSize = (bytes: number) => {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="/" aria-label="NitiSaathi home">
          <span className="brand-mark">N</span>
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
          {apiStatus === "checking"
            ? "Connecting"
            : apiStatus === "online"
              ? "System online"
              : "Backend offline"}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              GOVERNMENT POLICIES, MADE HUMAN
            </div>

            <h1>
              Understand your rights.
              <br />
              <span>Know your next step.</span>
            </h1>

            <p className="hero-description">
              Government policies can be complicated. We make them easier
              to understand with simple explanations in your language,
              backed by the original document.
            </p>

            <div className="hero-tags">
              <span>English</span>
              <span>हिन्दी</span>
              <span>Source-backed answers</span>
            </div>
          </div>

          <aside className="trust-card">
            <div className="trust-icon">✓</div>
            <h3>Understand with confidence</h3>
            <p>
              Every important explanation should lead back to the original
              policy and its exact page.
            </p>
            <div className="trust-divider" />
            <div className="trust-point">
              <span>01</span>
              Evidence before answers
            </div>
            <div className="trust-point">
              <span>02</span>
              Clear, simple language
            </div>
            <div className="trust-point">
              <span>03</span>
              Honest about uncertainty
            </div>
          </aside>
        </section>

        <section className="workspace" id="how-it-works">
          <div className="section-heading">
            <div>
              <span className="section-eyebrow">GET STARTED</span>
              <h2>Let's understand a policy.</h2>
              <p>Start with a government scheme or policy document.</p>
            </div>
            <span className="step-label">TWO SIMPLE STEPS</span>
          </div>

          <div className="workspace-grid">
            <section className="panel document-panel">
              <div className="panel-heading">
                <span className="step-number">01</span>
                <div>
                  <h3>Choose a document</h3>
                  <p>Upload the government PDF you want to understand.</p>
                </div>
              </div>

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
                disabled={uploadStatus === "uploading"}
              >
                <span className="upload-icon">
                  {uploadStatus === "uploading"
                    ? "…"
                    : uploadStatus === "error"
                      ? "!"
                      : uploadedDocument
                        ? "✓"
                        : "↑"}
                </span>

                {file ? (
                  <>
                    <strong>{file.name}</strong>
                    <span>{formatFileSize(file.size)}</span>
                    <small>
                      {uploadStatus === "uploading"
                        ? "Uploading and processing…"
                        : uploadedDocument
                          ? "Uploaded · click to choose another PDF"
                          : uploadStatus === "error"
                            ? "Upload failed · click to try another PDF"
                            : "Click to choose another PDF"}
                    </small>
                  </>
                ) : (
                  <>
                    <strong>Upload a policy document</strong>
                    <span>Click to browse files on your device</span>
                    <small>PDF files only · Maximum size 20 MB</small>
                  </>
                )}
              </button>

              {uploadStatus === "uploading" && (
                <p className="action-status" role="status" aria-live="polite">
                  Uploading and processing your PDF. The first upload may take
                  longer while the search model loads.
                </p>
              )}
              {uploadedDocument && (
                <p className="action-status success" role="status">
                  Ready to ask · {uploadedDocument.page_count} pages ·{" "}
                  {uploadedDocument.indexed_chunks} searchable sections
                </p>
              )}
              {uploadError && (
                <p className="action-status error" role="alert">
                  {uploadError}
                </p>
              )}
              {uploadedDocument?.warnings?.map((warning) => (
                <p className="action-status warning" key={warning}>
                  {warning}
                </p>
              ))}

              <div className="privacy-note">
                <span>⌑</span>
                Your document is used only for this demonstration.
              </div>
            </section>

            <section className="panel question-panel">
              <div className="panel-heading">
                <span className="step-number">02</span>
                <div>
                  <h3>Ask what matters to you</h3>
                  <p>Choose a language and start with a practical question.</p>
                </div>
              </div>

              <label className="field-label" htmlFor="language">
                EXPLANATION LANGUAGE
              </label>

              <select
                id="language"
                className="language-select"
                value={language}
                onChange={(event) => {
                  setLanguage(event.target.value);
                  setAnswer(null);
                  setAskError("");
                }}
              >
                <option value="Hindi">हिन्दी (Hindi)</option>
                <option value="English">English</option>
              </select>

              <label className="field-label question-label" htmlFor="question">
                WHAT WOULD YOU LIKE TO KNOW?
              </label>

              <textarea
                id="question"
                className="question-input"
                placeholder={
                  language === "Hindi"
                    ? "उदाहरण: क्या मैं इस योजना के लिए पात्र हूँ?"
                    : "For example: Am I eligible for this scheme?"
                }
                value={question}
                maxLength={1000}
                onChange={(event) => {
                  setQuestion(event.target.value);
                  setAnswer(null);
                  setAskError("");
                }}
                rows={3}
              />

              <div className="sample-heading">NOT SURE WHERE TO START?</div>

              <div className="sample-questions">
                {sampleQuestions.map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => {
                      setQuestion(sample);
                      setAnswer(null);
                      setAskError("");
                    }}
                    className="sample-chip"
                  >
                    {sample}
                    <span>↗</span>
                  </button>
                ))}
              </div>

              <button
                className="ask-button"
                type="button"
                disabled={
                  !uploadedDocument || !question.trim() || askStatus === "asking"
                }
                onClick={() => void handleAsk()}
              >
                {askStatus === "asking" ? "Finding an answer…" : "Ask about this policy"}
                <span>→</span>
              </button>

              {!uploadedDocument && !uploadError && (
                <p className="coming-soon">
                  Upload a PDF to enable questions about its contents.
                </p>
              )}
              {askError && (
                <p className="action-status error" role="alert">
                  {askError}
                </p>
              )}
              {answer && (
                <section className="answer-card" aria-live="polite">
                  <div className="answer-heading">
                    <div>
                      <span className="section-eyebrow">ANSWER</span>
                      <h4>Here’s what the policy says</h4>
                    </div>
                    <span className="answer-language">{answer.language}</span>
                  </div>
                  <p className="answer-text">{answer.answer}</p>
                  <div className="answer-sources">
                    <strong>Sources</strong>
                    {answer.sources.length > 0 ? (
                      <ul>
                        {answer.sources.map((source, index) => (
                          <li key={`${source.filename}-${source.page_number}-${index}`}>
                            {source.filename} · page {source.page_number}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>No source passages were returned for this answer.</p>
                    )}
                  </div>
                </section>
              )}
            </section>
          </div>
        </section>

        <section className="bottom-note" id="trust">
          <span>✳</span>
          <p>
            NitiSaathi explains policy documents; it does not provide legal
            advice or official government eligibility decisions.
          </p>
        </section>
      </main>

      <footer className="footer">
        <span>© 2026 NitiSaathi</span>
        <span>Built to make public information more accessible.</span>
      </footer>
    </div>
  );
}

export default App;

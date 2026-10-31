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

function App() {
  const [language, setLanguage] = useState("Hindi");
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
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

  // Handle PDF selection.
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (
      selectedFile.type !== "application/pdf" &&
      !selectedFile.name.toLowerCase().endsWith(".pdf")
    ) {
      alert("Please select a PDF document.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
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
              >
                <span className="upload-icon">
                  {file ? "✓" : "↑"}
                </span>

                {file ? (
                  <>
                    <strong>{file.name}</strong>
                    <span>{formatFileSize(file.size)}</span>
                    <small>Click to choose another PDF</small>
                  </>
                ) : (
                  <>
                    <strong>Upload a policy document</strong>
                    <span>Click to browse files on your device</span>
                    <small>PDF files only · Maximum size to be configured</small>
                  </>
                )}
              </button>

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
                onChange={(event) => setLanguage(event.target.value)}
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
                onChange={(event) => setQuestion(event.target.value)}
                rows={3}
              />

              <div className="sample-heading">NOT SURE WHERE TO START?</div>

              <div className="sample-questions">
                {sampleQuestions.map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => setQuestion(sample)}
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
                disabled
                title="Question answering will be enabled in the next milestone"
              >
                Ask about this policy <span>→</span>
              </button>

              <p className="coming-soon">
                Document processing and AI answers are our next milestone.
              </p>
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
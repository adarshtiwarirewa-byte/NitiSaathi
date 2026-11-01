# NitiSaathi Web Research Mode

## Behaviour

When no PDF is selected, the frontend calls \`POST /research/ask\`.
When a PDF is selected, the existing \`POST /documents/ask\` PDF-RAG path
continues unchanged. Uploading a PDF selects it and returns the user to the
document-grounded path.

The web research feature is isolated in:
- \`backend/app/api/research.py\`
- \`backend/app/services/web_research_service.py\`
- \`backend/tests/test_web_research_service.py\`

It does not write to ChromaDB, modify PDF ingestion, or change the existing
document endpoints.

## Configuration

Web Research uses the existing \`GEMINI_API_KEY\` and \`GEMINI_MODEL\`.
The configured Gemini model must support the Google Search grounding tool.
No separate search API key is required.

Sarvam remains the configured primary structured-answer provider. If the
existing LLM provider is configured as Gemini, the grounded evidence is sent
to Gemini for structured explanation. Sarvam/Gemini output is constrained to
the retrieved source passages.

## Source policy

- HTTPS government domains under \`.gov.in\` and \`.nic.in\`, plus PIB, are
  labelled official-government sources.
- A small allowlist of established Indian publications may be used as
  supporting context, never as official government authority.
- Sources without Google grounding support are excluded.
- Every answer point must cite one or more retained source numbers.
- Retrieval timestamps are recorded. Publication dates are not claimed unless
  they are supplied by the search provider.

## API

\`POST /research/ask\`

\`\`\`json
{
  "question": "Who is eligible for PM-KISAN?",
  "language": "English",
  "top_k": 6
}
\`\`\`

Returns \`source_mode: "web_research"\`, the seven existing policy explanation
sections, and web source metadata (\`title\`, \`url\`, \`domain\`,
\`source_type\`, \`retrieved_at\`).

## Limitations

Google Search grounding and the Gemini model must be enabled for the API key.
A search result is not itself proof that a scheme is currently active. Users
should follow the linked issuing authority for final, time-sensitive details.
If the search provider supplies no verifiable grounded passages, the endpoint
returns a clear service error instead of fabricating an answer.

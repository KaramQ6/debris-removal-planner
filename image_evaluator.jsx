import { useState, useCallback } from "react";

const MODEL = "claude-sonnet-4-20250514";

const EVAL_PROMPT = `You are reviewing orbital debris removal mission visualization images for a hackathon submission (AESS Sustainability Hackathon 2026, team ta5abes).

The project uses RL (PPO) + RAG for fuel-optimal debris collection planning. The images are 3D matplotlib orbital plots showing spacecraft trajectories around Earth with 8 debris targets.

Please evaluate this image and provide:

1. **Visual Quality** (1-10): Clarity, colors, readability
2. **Technical Content** (1-10): Does it clearly show the mission path, debris positions, delta-v info?
3. **Submission Readiness** (1-10): Is this suitable for a hackathon judges panel?
4. **What it shows**: Brief description of what policy/trajectory is displayed
5. **Strengths**: Top 2 visual/technical strengths
6. **Issues**: Any problems (clutter, poor contrast, missing labels, etc.)
7. **Fix Recommendation**: One concrete improvement if needed
8. **GitHub Alternative**: Since this file is too large for GitHub — should they (a) compress it, (b) use the HTML interactive version instead, or (c) drop it and rely on the 2D polar plot?

Be direct and specific. This is for a real competition submission.`;

export default function ImageEvaluator() {
  const [images, setImages] = useState([]);
  const [evaluations, setEvaluations] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const handleFiles = useCallback((files) => {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    setImages(prev => {
      const existing = new Set(prev.map(i => i.name));
      const newOnes = imageFiles.filter(f => !existing.has(f.name));
      return [...prev, ...newOnes.map(f => ({ file: f, name: f.name, url: URL.createObjectURL(f) }))];
    });
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onDragOver = (e) => e.preventDefault();

  const evaluateImage = async (img) => {
    setLoading(prev => ({ ...prev, [img.name]: true }));
    setError(null);
    try {
      const base64 = await new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result.split(",")[1]);
        reader.onerror = rej;
        reader.readAsDataURL(img.file);
      });

      const mediaType = img.file.type || "image/png";

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: 1000,
          messages: [{
            role: "user",
            content: [
              { type: "image", source: { type: "base64", media_type: mediaType, data: base64 } },
              { type: "text", text: EVAL_PROMPT }
            ]
          }]
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error?.message || `HTTP ${response.status}`);
      }

      const data = await response.json();
      const text = data.content?.find(b => b.type === "text")?.text || "No response";
      setEvaluations(prev => ({ ...prev, [img.name]: text }));
    } catch (e) {
      setError(`${img.name}: ${e.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [img.name]: false }));
    }
  };

  const evaluateAll = () => {
    images.forEach(img => {
      if (!evaluations[img.name] && !loading[img.name]) evaluateImage(img);
    });
  };

  const removeImage = (name) => {
    setImages(prev => prev.filter(i => i.name !== name));
    setEvaluations(prev => { const n = {...prev}; delete n[name]; return n; });
  };

  const anyLoading = Object.values(loading).some(Boolean);

  return (
    <div style={{ fontFamily: "'Segoe UI', sans-serif", background: "#0f1117", minHeight: "100vh", color: "#e2e8f0", padding: "24px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#7c3aed", margin: 0 }}>
            🛰️ Mission Plot Evaluator
          </h1>
          <p style={{ color: "#64748b", margin: "6px 0 0", fontSize: 13 }}>
            ta5abes — AESS Hackathon 2026 | Drop your large 3D PNGs here for AI evaluation
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onClick={() => document.getElementById("file-input").click()}
          style={{
            border: "2px dashed #334155",
            borderRadius: 12,
            padding: "40px 24px",
            textAlign: "center",
            cursor: "pointer",
            marginBottom: 20,
            background: "#1e293b",
            transition: "border-color 0.2s"
          }}
        >
          <div style={{ fontSize: 36, marginBottom: 8 }}>📁</div>
          <div style={{ color: "#94a3b8", fontSize: 14 }}>
            اسحب mission_3d_*.png هنا أو اضغط للاختيار
          </div>
          <div style={{ color: "#475569", fontSize: 12, marginTop: 4 }}>
            Supports PNG, JPG — any size
          </div>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            multiple
            style={{ display: "none" }}
            onChange={e => handleFiles(e.target.files)}
          />
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#7f1d1d", border: "1px solid #ef4444", borderRadius: 8, padding: "10px 14px", marginBottom: 16, fontSize: 13, color: "#fca5a5" }}>
            ⚠️ {error}
          </div>
        )}

        {/* Evaluate All Button */}
        {images.length > 0 && (
          <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
            <button
              onClick={evaluateAll}
              disabled={anyLoading}
              style={{
                background: anyLoading ? "#334155" : "#7c3aed",
                color: "#fff",
                border: "none",
                borderRadius: 8,
                padding: "10px 20px",
                fontSize: 14,
                fontWeight: 600,
                cursor: anyLoading ? "not-allowed" : "pointer"
              }}
            >
              {anyLoading ? "⏳ Evaluating..." : `🤖 Evaluate All (${images.length})`}
            </button>
            <button
              onClick={() => { setImages([]); setEvaluations({}); }}
              style={{ background: "transparent", color: "#64748b", border: "1px solid #334155", borderRadius: 8, padding: "10px 16px", fontSize: 13, cursor: "pointer" }}
            >
              Clear
            </button>
          </div>
        )}

        {/* Image Cards */}
        {images.map(img => (
          <div key={img.name} style={{ background: "#1e293b", borderRadius: 12, marginBottom: 20, overflow: "hidden", border: "1px solid #334155" }}>
            {/* Card Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid #334155" }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: 14, color: "#e2e8f0" }}>{img.name}</span>
                <span style={{ marginLeft: 10, color: "#64748b", fontSize: 12 }}>
                  {(img.file.size / 1024 / 1024).toFixed(1)} MB
                </span>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={() => evaluateImage(img)}
                  disabled={loading[img.name]}
                  style={{
                    background: loading[img.name] ? "#334155" : "#1d4ed8",
                    color: "#fff",
                    border: "none",
                    borderRadius: 6,
                    padding: "6px 12px",
                    fontSize: 12,
                    cursor: loading[img.name] ? "not-allowed" : "pointer"
                  }}
                >
                  {loading[img.name] ? "⏳..." : "Evaluate"}
                </button>
                <button
                  onClick={() => removeImage(img.name)}
                  style={{ background: "transparent", color: "#ef4444", border: "1px solid #ef4444", borderRadius: 6, padding: "6px 10px", fontSize: 12, cursor: "pointer" }}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Image Preview */}
            <div style={{ display: "flex", gap: 0 }}>
              <div style={{ width: 240, minWidth: 240, background: "#0f1117", display: "flex", alignItems: "center", justifyContent: "center", padding: 12 }}>
                <img
                  src={img.url}
                  alt={img.name}
                  style={{ maxWidth: "100%", maxHeight: 180, objectFit: "contain", borderRadius: 6 }}
                />
              </div>

              {/* Evaluation Output */}
              <div style={{ flex: 1, padding: 16 }}>
                {loading[img.name] && (
                  <div style={{ color: "#7c3aed", fontSize: 13 }}>
                    🤖 Claude is reviewing the mission plot...
                  </div>
                )}
                {!loading[img.name] && !evaluations[img.name] && (
                  <div style={{ color: "#475569", fontSize: 13 }}>
                    Click "Evaluate" to get AI analysis of this plot
                  </div>
                )}
                {evaluations[img.name] && (
                  <div style={{ fontSize: 13, lineHeight: 1.7, color: "#cbd5e1", whiteSpace: "pre-wrap" }}>
                    {evaluations[img.name]}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Summary Section */}
        {Object.keys(evaluations).length >= 2 && (
          <SummaryPanel evaluations={evaluations} images={images} />
        )}

        {images.length === 0 && (
          <div style={{ textAlign: "center", color: "#475569", marginTop: 40, fontSize: 13 }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}>🛸</div>
            <div>No images loaded yet</div>
            <div style={{ marginTop: 8, color: "#334155" }}>
              mission_3d_nearest.png · mission_3d_random.png · mission_3d_risk_weighted.png
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SummaryPanel({ evaluations, images }) {
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  const generateSummary = async () => {
    setLoading(true);
    const combined = Object.entries(evaluations)
      .map(([name, eval_]) => `=== ${name} ===\n${eval_}`)
      .join("\n\n");

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: 600,
          messages: [{
            role: "user",
            content: `Based on these individual image evaluations for a hackathon submission, give a 3-paragraph summary:
1. Overall visual quality verdict across all plots
2. Which plot is strongest and why
3. GitHub strategy recommendation (which files to commit, which to reference as HTML interactive instead)

Keep it direct and actionable. Context: these are orbital debris removal mission path visualizations.

${combined}`
          }]
        })
      });
      const data = await res.json();
      setSummary(data.content?.find(b => b.type === "text")?.text || "");
    } catch (e) {
      setSummary("Error generating summary: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: "#0f172a", border: "1px solid #7c3aed", borderRadius: 12, padding: 20, marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <h3 style={{ margin: 0, fontSize: 15, color: "#a78bfa" }}>📊 Overall Summary & GitHub Strategy</h3>
        <button
          onClick={generateSummary}
          disabled={loading}
          style={{ background: loading ? "#334155" : "#7c3aed", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, cursor: loading ? "not-allowed" : "pointer" }}
        >
          {loading ? "⏳ Generating..." : "Generate Summary"}
        </button>
      </div>
      {summary && (
        <div style={{ fontSize: 13, lineHeight: 1.8, color: "#cbd5e1", whiteSpace: "pre-wrap" }}>
          {summary}
        </div>
      )}
      {!summary && !loading && (
        <div style={{ color: "#475569", fontSize: 13 }}>
          اضغط "Generate Summary" لتجميع نتائج كل الصور في تقرير واحد مع توصية GitHub
        </div>
      )}
    </div>
  );
}

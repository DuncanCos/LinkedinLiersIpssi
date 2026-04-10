import { useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

function App() {
  const [text, setText] = useState('')
  const [summary, setSummary] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setSummary('')
    setPrediction(null)

    if (!text.trim()) {
      setError('Entre un texte avant de lancer l\'analyse.')
      return
    }

    setLoading(true)

    try {
      const [summaryRes, classifyRes] = await Promise.all([
        fetch(`${API_BASE_URL}/summaries`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        }),
        fetch(`${API_BASE_URL}/classify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })
      ])

      const summaryPayload = await summaryRes.json()
      const classifyPayload = await classifyRes.json()

      if (!summaryRes.ok) {
        throw new Error(summaryPayload?.detail || 'Erreur serveur pour le résumé.')
      }
      if (!classifyRes.ok) {
        throw new Error(classifyPayload?.detail || 'Erreur serveur pour la classification.')
      }

      setSummary(summaryPayload.summary || '')
      
      const value = Number(classifyPayload?.prediction)
      if (value !== 0 && value !== 1) {
        throw new Error('Réponse de classification invalide.')
      }
      setPrediction(value)
    } catch (submitError) {
      setError(submitError.message || 'Impossible de joindre le backend.')
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setText('')
    setSummary('')
    setPrediction(null)
    setError('')
  }

  return (
    <main className="app-container">
  <div className="glass-card">
    <header className="header">
      <div className="logo">LL</div>
      <h1>LinkedIn <span className="highlight">Liers</span></h1>
      <p className="subtitle">L'IA qui dégonfle les posts trop longs.</p>
    </header>

    <form className="form" onSubmit={handleSubmit}>
      <div className="input-group">
        <label htmlFor="text">Texte à résumer</label>
        <textarea
          id="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={8}
          placeholder="Colle ici le post de 50 lignes du Zack..."
          required
        />
        <div className="char-count">{text.length} caractères</div>
      </div>

      <div className="form-actions">
        <button type="submit" className={`submit-btn ${loading ? 'loading' : ''}`} disabled={loading || !text}>
          {loading ? (
            <span className="spinner"></span>
          ) : (
            'Analyser le post'
          )}
        </button>
        <button type="button" className="clear-btn" onClick={handleClear} disabled={loading || !text}>
          Clean
        </button>
      </div>
    </form>

    {error && (
      <div className="error-badge">
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" width="16"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        {error}
      </div>
    )}

    {summary && (
      <section className="result-area animate-in">
        <div className="result-header">
          <h2>Résumé exécutif</h2>
          <button onClick={() => navigator.clipboard.writeText(summary)} className="copy-btn">
            Copier
          </button>
        </div>
        <div className="summary-content">
          <p>{summary}</p>
        </div>
      </section>
    )}

    {prediction !== null && (
      <section className="result-area animate-in">
        <div className="result-header">
          <h2>Classification</h2>
        </div>
        <div className={`classification-content ${prediction === 1 ? 'classification-slop' : 'classification-ok'}`}>
          <p className="prediction-value">{prediction}</p>
          <p className="prediction-label">
            {prediction === 1 ? 'TOTAL IA SLOP' : 'isOK'}
          </p>
        </div>
      </section>
    )}
  </div>
</main>
  )
}

export default App

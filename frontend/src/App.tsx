
import React, { useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

function App() {
  const [loading, setLoading] = useState(false)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [conf, setConf] = useState<number>(0.4)
  const [imgsz, setImgsz] = useState<number>(640)
  const [margin, setMargin] = useState<number>(15)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)

  async function uploadFile(file: File) {
    setLoading(true)
    setVideoUrl(null)
    try {
      const fd = new FormData()
      fd.append('video', file, file.name)
      // append processing options
      fd.append('conf', String(conf))
      fd.append('imgsz', String(imgsz))
      fd.append('margin', String(margin))

      const res = await fetch('http://localhost:5000/upload', { method: 'POST', body: fd })
      if (!res.ok) throw new Error('Upload failed: ' + res.statusText)
      const j = await res.json()
      const jobId = j.job_id
      // poll status
      let finalUrl: string | null = null
      while (true) {
        await new Promise(r => setTimeout(r, 1500))
        const sres = await fetch(`http://localhost:5000/status/${jobId}`)
        if (!sres.ok) throw new Error('Status failed')
        const sj = await sres.json()
        if (sj.status === 'processing' || sj.status === 'queued') {
          setLoading(true)
          if (typeof sj.progress === 'number') {
            setProgress(Number(sj.progress))
          }
          continue
        }
        if (sj.status === 'done') {
          const fres = await fetch(`http://localhost:5000/result/${jobId}`)
          if (!fres.ok) throw new Error('Failed to fetch result')
          const blob = await fres.blob()
          finalUrl = URL.createObjectURL(blob)
          break
        }
        if (sj.status === 'error') {
          throw new Error(sj.error || 'Processing error')
        }
        // unknown status
        break
      }
      if (finalUrl) setVideoUrl(finalUrl)
    } catch (err: any) {
      alert(err?.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files && e.target.files[0]
    if (f) setSelectedFile(f)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0])
  }

  const onDragOver = (e: React.DragEvent) => e.preventDefault()

  return (
    <>
      <main style={{ padding: 20 }}>
        <h2>Wgraj swój filmik z parkingu</h2>

        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          style={{ border: '2px dashed #ccc', padding: 20, borderRadius: 8 }}>
          <p>Przeciągnij plik tutaj lub wybierz poniżej.</p>
          <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
            <label>
              Confidence:
              <input type="number" step="0.01" min="0" max="1" value={conf}
                onChange={e => setConf(parseFloat(e.target.value) || 0.0)} style={{ marginLeft: 6, width: 90 }} />
            </label>
            <label>
              Img size:
              <input type="number" step="1" min="64" value={imgsz}
                onChange={e => setImgsz(parseInt(e.target.value) || 640)} style={{ marginLeft: 6, width: 100 }} />
            </label>
            <label>
              Margin:
              <input type="number" step="1" min="0" value={margin}
                onChange={e => setMargin(parseInt(e.target.value) || 15)} style={{ marginLeft: 6, width: 80 }} />
            </label>
          </div>
          <input ref={fileRef} type="file" accept="video/*" onChange={onFileChange} />
          <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
            <button onClick={() => selectedFile && uploadFile(selectedFile)} disabled={!selectedFile || loading}>
              Start processing
            </button>
            <div style={{ fontSize: 13, color: '#333' }}>{selectedFile ? `Selected: ${selectedFile.name}` : 'No file chosen'}</div>
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          {loading && (
            <div>
              <div>Loading... Przetwarzanie trwa — proszę czekać.</div>
              {progress !== null && (
                <div style={{ width: '100%', background: '#eee', borderRadius: 4, marginTop: 8 }}>
                  <div style={{ width: `${progress}%`, background: '#4caf50', height: 12, borderRadius: 4 }} />
                </div>
              )}
              {progress !== null && <div style={{ marginTop: 6 }}>{progress}%</div>}
            </div>
          )}

          {videoUrl && (
            <div>
              <h3>Wygenerowany film</h3>
              <video id="resultVideo" src={videoUrl} controls style={{ maxWidth: '100%' }} />
              <div style={{ marginTop: 8 }}>
                <a href={videoUrl} download="out.mp4">
                  <button>Download result</button>
                </a>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  )
}

export default App

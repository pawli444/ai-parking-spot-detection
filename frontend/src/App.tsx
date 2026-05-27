import React, { useRef, useState } from 'react'
import './App.css'

function App() {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)

  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [csvUrl, setCsvUrl] = useState<string | null>(null)
  const [plotUrl, setPlotUrl] = useState<string | null>(null)

  const [conf, setConf] = useState<number>(0.2)
  const [imgsz, setImgsz] = useState<number>(960)
  const [iou, setIou] = useState<number>(0.6)
  const [device, setDevice] = useState<string>('0')
  const [saveFlag, setSaveFlag] = useState<boolean>(false)
  const [margin, setMargin] = useState<number>(15)

  // osobne stany na dwa rozne pliki
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null)
  const [selectedSpots, setSelectedSpots] = useState<File | null>(null)

  const videoRef = useRef<HTMLInputElement | null>(null)
  const spotsRef = useRef<HTMLInputElement | null>(null)

  async function uploadFile() {
    if (!selectedVideo || !selectedSpots) return

    setLoading(true)
    setVideoUrl(null)
    setCsvUrl(null)
    setPlotUrl(null)

    try {
      const fd = new FormData()
      fd.append('video', selectedVideo, selectedVideo.name)
      // dodajemy jsona do pakiety
      fd.append('spots', selectedSpots, selectedSpots.name)

      fd.append('conf', String(conf))
      fd.append('imgsz', String(imgsz))
      fd.append('iou', String(iou))
      fd.append('device', device)
      fd.append('save', String(saveFlag))
      fd.append('margin', String(margin))

      const res = await fetch('http://localhost:5000/upload', { method: 'POST', body: fd })
      if (!res.ok) throw new Error('Upload failed: ' + res.statusText)
      const j = await res.json()
      const jobId = j.job_id

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
          setVideoUrl(`http://localhost:5000/result/${jobId}/video`)
          setCsvUrl(`http://localhost:5000/result/${jobId}/csv`)
          setPlotUrl(`http://localhost:5000/result/${jobId}/plot`)
          break
        }

        if (sj.status === 'error') {
          throw new Error(sj.error || 'Processing error')
        }
        break
      }
    } catch (err: any) {
      alert(err?.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  // handlery inputow
  const onVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files && e.target.files[0]
    if (f) setSelectedVideo(f)
  }

  const onSpotsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files && e.target.files[0]
    if (f) setSelectedSpots(f)
  }

  const onDropVideo = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setSelectedVideo(e.dataTransfer.files[0])
  }

  const onDragOver = (e: React.DragEvent) => e.preventDefault()

  return (
    <>
      <main style={{ padding: 20, maxWidth: '800px', margin: '0 auto' }}>
        <h2>Wgraj filmik i siatkę miejsc z parkingu</h2>

        <div
          onDrop={onDropVideo}
          onDragOver={onDragOver}
          style={{ border: '2px dashed #ccc', padding: 20, borderRadius: 8 }}>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginBottom: '20px', background: '#f9f9f9', padding: '15px', borderRadius: '8px' }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px' }}>
              <strong>imgsz:</strong>
              <input type="number" step="32" min="64" value={imgsz} onChange={e => setImgsz(parseInt(e.target.value) || 960)} style={{ width: 100 }} />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px' }}>
              <strong>conf:</strong>
              <input type="number" step="0.05" min="0" max="1" value={conf} onChange={e => setConf(parseFloat(e.target.value) || 0.2)} style={{ width: 100 }} />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px' }}>
              <strong>iou:</strong>
              <input type="number" step="0.05" min="0" max="1" value={iou} onChange={e => setIou(parseFloat(e.target.value) || 0.6)} style={{ width: 100 }} />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px' }}>
              <strong>device:</strong>
              <select value={device} onChange={e => setDevice(e.target.value)} style={{ width: 108 }}>
                <option value="0">GPU (0)</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px', alignItems: 'center' }}>
              <strong>save:</strong>
              <input type="checkbox" checked={saveFlag} onChange={e => setSaveFlag(e.target.checked)} />
            </label>
            <label style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '300px' }}>
              <strong>margin:</strong>
              <input type="number" step="1" min="0" value={margin} onChange={e => setMargin(parseInt(e.target.value) || 15)} style={{ width: 100 }} />
            </label>
          </div>

          <div style={{ marginBottom: '20px', padding: '10px', background: '#eef2f5', borderRadius: '8px' }}>
            <p style={{ margin: '0 0 10px 0', fontWeight: 'bold' }}>1. Plik wideo (.mp4)</p>
            <input ref={videoRef} type="file" accept="video/*" onChange={onVideoChange} style={{ display: 'block' }} />
            <div style={{ fontSize: 13, color: '#555', marginTop: 5 }}>
              {selectedVideo ? `Wybrano wideo: ${selectedVideo.name}` : 'Brak pliku wideo'}
            </div>
          </div>

          <div style={{ marginBottom: '20px', padding: '10px', background: '#eef2f5', borderRadius: '8px' }}>
            <p style={{ margin: '0 0 10px 0', fontWeight: 'bold' }}>2. Plik konfiguracyjny miejsc (.json)</p>
            <input ref={spotsRef} type="file" accept=".json" onChange={onSpotsChange} style={{ display: 'block' }} />
            <div style={{ fontSize: 13, color: '#555', marginTop: 5 }}>
              {selectedSpots ? `Wybrano plik miejsc: ${selectedSpots.name}` : 'Brak pliku z miejscami'}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-start', marginTop: '10px' }}>
            {/* przycisk zablokowany jesli brakuje wideo ALBO jsona */}
            <button
              onClick={uploadFile}
              disabled={!selectedVideo || !selectedSpots || loading}
              style={{ padding: '12px 25px', cursor: (!selectedVideo || !selectedSpots) ? 'not-allowed' : 'pointer', background: (!selectedVideo || !selectedSpots) ? '#ccc' : '#4caf50', color: '#fff', border: 'none', borderRadius: '5px', fontWeight: 'bold' }}>
              START
            </button>
            {(!selectedVideo || !selectedSpots) && (
              <span style={{color: 'red', fontSize: '13px'}}>Musisz dodać oba pliki żeby wystartować.</span>
            )}
          </div>
        </div>

        <div style={{ marginTop: 20 }}>
          {loading && (
            <div style={{ padding: '15px', background: '#fff8e1', borderRadius: '8px', border: '1px solid #ffe082' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>Przetwarzanie trwa — proszę czekać...</div>
              {progress !== null && (
                <div style={{ width: '100%', background: '#eee', borderRadius: 4 }}>
                  <div style={{ width: `${progress}%`, background: '#4caf50', height: 16, borderRadius: 4, transition: 'width 0.3s' }} />
                </div>
              )}
              {progress !== null && <div style={{ marginTop: 6, fontWeight: 'bold' }}>{progress}%</div>}
            </div>
          )}

          {videoUrl && (
            <div style={{ marginTop: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
              <h3>Wyniki analizy</h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', marginTop: '20px' }}>

                <div>
                  <h4 style={{ margin: '0 0 10px 0' }}>Gotowy film</h4>
                  <video id="resultVideo" src={videoUrl} controls style={{ width: '100%', borderRadius: '8px', background: '#000' }} />
                  <div style={{ marginTop: 10 }}>
                    <a href={videoUrl} download="out.mp4">
                      <button style={{ padding: '8px 16px' }}>Pobierz Wideo (MP4)</button>
                    </a>
                  </div>
                </div>

                {plotUrl && (
                  <div>
                    <h4 style={{ margin: '0 0 10px 0' }}>Wykres trendu zajętości</h4>
                    <img src={plotUrl} alt="Wykres zajętości parkingu" style={{ width: '100%', borderRadius: '8px', border: '1px solid #eee' }} />
                    <div style={{ marginTop: 10 }}>
                      <a href={plotUrl} download="wykres.png" target="_blank" rel="noreferrer">
                        <button style={{ padding: '8px 16px' }}>Pobierz Wykres (PNG)</button>
                      </a>
                    </div>
                  </div>
                )}

                {csvUrl && (
                  <div style={{ padding: '20px', background: '#f0f8ff', borderRadius: '8px', border: '1px solid #cce0ff' }}>
                    <h4 style={{ margin: '0 0 10px 0' }}>Logi danych</h4>
                    <p style={{ fontSize: '14px', margin: '0 0 15px 0' }}>Szczegółowe dane klatka po klatce w formacie arkusza kalkulacyjnego.</p>
                    <a href={csvUrl} download="logi.csv">
                      <button style={{ padding: '10px 20px', background: '#0066cc', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
                        Pobierz dane (CSV)
                      </button>
                    </a>
                  </div>
                )}

              </div>
            </div>
          )}
        </div>
      </main>
    </>
  )
}

export default App
import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { Moon, Sun, BarChart2, Upload, Columns, BrainCircuit, CheckCircle2, ChevronDown, ChevronRight } from 'lucide-react'

const REGRESSION_MODELS = [
  { key: 'decision_tree_regressor', label: 'Arbol & Random Forest', desc: 'Ensemble con bagging' },
  { key: 'linear_regression',       label: 'Regresion Lineal',      desc: 'Minimos cuadrados OLS' },
  { key: 'polynomial_regression',   label: 'Polinomial',            desc: 'Features cuadraticas grado 2' },
  { key: 'ridge_lasso',             label: 'Ridge & Lasso',         desc: 'Regularizacion L1 y L2' },
  { key: 'svr',                     label: 'SVR',                   desc: 'Support Vector Regression' },
  { key: 'knn_regressor',           label: 'KNN Regressor',         desc: 'K=5 vecinos mas cercanos' },
]

const CLASSIFICATION_MODELS = [
  { key: 'logistic_regression',      label: 'Regresion Logistica', desc: 'Funcion sigmoide' },
  { key: 'svm',                      label: 'SVM',                 desc: 'Margen maximo, kernel RBF' },
  { key: 'knn_classifier',           label: 'KNN',                 desc: 'K=5 vecinos mas cercanos' },
  { key: 'decision_tree_classifier', label: 'Arbol',               desc: 'Profundidad maxima 4' },
  { key: 'random_forest_classifier', label: 'Random Forest',       desc: 'Ensemble de 150 arboles' },
  { key: 'extra_trees',              label: 'Extra Trees',         desc: '100 arboles aleatorios' },
  { key: 'ridge_classifier',         label: 'Ridge Classifier',    desc: 'Regularizacion L2' },
  { key: 'qda',                      label: 'QDA',                 desc: 'Analisis discriminante cuadratico' },
  { key: 'naive_bayes',              label: 'Naive Bayes',         desc: 'Probabilidad gaussiana' },
  { key: 'gradient_boosting',        label: 'Gradient Boosting',   desc: 'Arboles secuenciales' },
  { key: 'adaboost',                 label: 'AdaBoost',            desc: 'Ponderacion adaptativa' },
  { key: 'perceptron',               label: 'Perceptron',          desc: 'Neurona lineal binaria' },
  { key: 'mlp_classifier',           label: 'MLP',                 desc: 'Red neuronal 10x10' },
]

const PAGE_SIZES = [10, 25, 50, 100]

function extractMetric(result) {
  if (!result) return null
  if (result.type === 'regression') {
    const r2 = result.metrics.r2 ?? result.metrics.r2_arbol ?? result.metrics.r2_ridge
    if (r2 === undefined) return null
    return { tag: 'R2', display: r2.toFixed(4), raw: r2 }
  }
  const acc = result.metrics.accuracy
  if (acc === undefined) return null
  return { tag: 'Exactitud', display: acc.toFixed(2) + '%', raw: acc }
}

function Spinner({ invert }) {
  return (
    <div className={`w-4 h-4 rounded-full border-2 animate-spin flex-shrink-0
      ${invert ? 'border-black/20 border-t-black' : 'border-gray-300 border-t-gray-600 dark:border-white/20 dark:border-t-white'}`} />
  )
}

function ThemeToggle({ dark, onToggle }) {
  return (
    <button onClick={onToggle}
      className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all
        border-gray-200 bg-gray-50 hover:bg-gray-100 text-gray-600
        dark:border-white/10 dark:bg-white/5 dark:hover:bg-white/10 dark:text-gray-300">
      {dark ? <Sun size={13} /> : <Moon size={13} />}
      {dark ? 'Modo dia' : 'Modo oscuro'}
    </button>
  )
}

function SectionLabel({ children }) {
  return <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3">{children}</p>
}

function ColPill({ label, selected, onClick }) {
  return (
    <button onClick={onClick}
      className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all
        ${selected
          ? 'bg-gray-900 text-white border-gray-900 dark:bg-white dark:text-black dark:border-white'
          : 'text-gray-500 border-gray-200 hover:border-gray-400 hover:text-gray-700 dark:text-gray-400 dark:border-gray-700 dark:hover:border-gray-500 dark:hover:text-gray-200'}`}>
      {label}
    </button>
  )
}

function Tag({ label, onRemove }) {
  return (
    <span className="inline-flex items-center gap-1.5 pl-2.5 pr-1.5 py-1 rounded-lg text-xs font-semibold
      bg-blue-50 text-blue-700 border border-blue-100
      dark:bg-blue-500/10 dark:text-blue-300 dark:border-blue-500/20">
      {label}
      {onRemove && (
        <button onClick={onRemove}
          className="w-4 h-4 rounded flex items-center justify-center hover:bg-blue-200 dark:hover:bg-blue-500/30 transition-colors text-blue-400">
          x
        </button>
      )}
    </span>
  )
}

function ModelCard({ model, color, active, loading, result, onClick }) {
  const metric = extractMetric(result)
  return (
    <button onClick={onClick} disabled={loading}
      className={`relative rounded-2xl border p-4 text-left transition-all duration-150
        ${active
          ? 'bg-gray-900 border-gray-900 dark:bg-white dark:border-white shadow-lg'
          : 'border-gray-200 bg-white hover:border-gray-400 dark:border-gray-800 dark:bg-[#141414] dark:hover:border-gray-600 dark:hover:bg-[#1c1c1c] shadow-sm'}`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`w-2 h-2 rounded-full ${active ? 'opacity-30 bg-white dark:bg-black' : color}`} />
        {active && loading && <Spinner invert={false} />}
        {metric && !loading && (
          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded
            ${active ? 'bg-white/10 text-white dark:bg-black/10 dark:text-black/60' : 'bg-gray-100 text-gray-500 dark:bg-white/5 dark:text-gray-400'}`}>
            {metric.display}
          </span>
        )}
      </div>
      <p className={`font-semibold text-sm leading-tight mb-1 ${active ? 'text-white dark:text-black' : 'text-gray-800 dark:text-gray-100'}`}>
        {model.label}
      </p>
      <p className={`text-xs leading-relaxed ${active ? 'text-white/60 dark:text-black/40' : 'text-gray-400 dark:text-gray-500'}`}>
        {model.desc}
      </p>
    </button>
  )
}

function BigNum({ label, value }) {
  return (
    <div className="text-right">
      <p className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-1">{label}</p>
      <p className="text-5xl font-black tracking-tighter text-gray-900 dark:text-white">{value}</p>
    </div>
  )
}

function MetricBox({ label, value }) {
  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-[#111] px-4 py-3">
      <p className="text-xs font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-bold text-gray-900 dark:text-white tabular-nums">{value}</p>
    </div>
  )
}

function ProbBar({ label, pct }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-500 dark:text-gray-400 w-28 truncate text-right">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-gray-100 dark:bg-white/10 overflow-hidden">
        <div className="h-full rounded-full bg-blue-500" style={{ width: Math.min(100, Math.round(pct * 100)) + '%' }} />
      </div>
      <span className="text-xs font-bold text-gray-600 dark:text-gray-400 w-14 tabular-nums text-right">
        {(pct * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function DropZone({ file, onFile }) {
  const [drag, setDrag] = useState(false)
  const ref = useRef()
  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f) }}
      onClick={() => ref.current.click()}
      className={`cursor-pointer rounded-2xl border-2 border-dashed transition-all p-8 text-center
        ${drag ? 'border-blue-400 bg-blue-50 dark:bg-blue-500/10' : 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500'}`}>
      <input ref={ref} type="file" accept=".csv" className="hidden" onChange={e => { if (e.target.files[0]) onFile(e.target.files[0]) }} />
      {file ? (
        <div>
          <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 truncate">{file.name}</p>
          <p className="text-xs text-gray-400 dark:text-gray-600 mt-1">{(file.size / 1024).toFixed(1)} KB — clic para cambiar</p>
        </div>
      ) : (
        <div>
          <div className="w-12 h-12 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1a1a1a] flex items-center justify-center mx-auto mb-4">
            <Upload size={20} className="text-gray-400" />
          </div>
          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">Arrastra un CSV</p>
          <p className="text-xs text-gray-400 dark:text-gray-600 mt-1">o haz clic para seleccionar</p>
        </div>
      )}
    </div>
  )
}

function PrimaryBtn({ children, onClick, disabled, loading, full }) {
  return (
    <button onClick={onClick} disabled={disabled || loading}
      className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all
        bg-gray-900 text-white hover:bg-gray-700 dark:bg-white dark:text-black dark:hover:bg-gray-200
        disabled:opacity-40 disabled:cursor-not-allowed ${full ? 'w-full' : ''}`}>
      {loading && <Spinner invert />}
      {children}
    </button>
  )
}

function SecondaryBtn({ children, onClick, disabled, loading }) {
  return (
    <button onClick={onClick} disabled={disabled || loading}
      className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#141414] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-[#1c1c1c] transition-all disabled:opacity-40 disabled:cursor-not-allowed">
      {loading && <Spinner />}
      {children}
    </button>
  )
}

export default function App() {
  const [dark, setDark]                       = useState(false)
  const [tab, setTab]                         = useState('modelos')
  const [sessionId, setSessionId]             = useState(null)
  const [step, setStep]                       = useState(1)
  const [file, setFile]                       = useState(null)
  const [fileData, setFileData]               = useState(null)
  const [uploading, setUploading]             = useState(false)
  const [xCols, setXCols]                     = useState([])
  const [yCol, setYCol]                       = useState('')
  const [activeModel, setActiveModel]         = useState(null)
  const [modelResults, setModelResults]       = useState({})
  const [training, setTraining]               = useState(false)
  const [trainResult, setTrainResult]         = useState(null)
  const [trainError, setTrainError]           = useState(null)
  const [predValues, setPredValues]           = useState({})
  const [predicting, setPredicting]           = useState(false)
  const [prediction, setPrediction]           = useState(null)
  const [predError, setPredError]             = useState(null)
  const [detailedMetrics, setDetailedMetrics] = useState(null)
  const [loadingMetrics, setLoadingMetrics]   = useState(false)
  const [history, setHistory]                 = useState([])
  const [modelPredictions, setModelPredictions] = useState({})
  const [currentPage, setCurrentPage]         = useState(1)
  const [pageSize, setPageSize]               = useState(10)
  const [previewRows, setPreviewRows]         = useState([])
  const [totalRows, setTotalRows]             = useState(0)
  const [loadingPreview, setLoadingPreview]   = useState(false)
  const [expandedIds, setExpandedIds]          = useState(new Set())
  const resultsRef = useRef(null)
  const activeHistoryId = useRef(null)

  useEffect(() => { document.documentElement.classList.toggle('dark', dark) }, [dark])
  useEffect(() => {
    axios.post('/api/session')
      .then(r => setSessionId(r.data.session_id))
      .catch(() => alert('No se pudo conectar con el servidor. Asegurate de que el backend este corriendo.'))
  }, [])

  const resetToStep1 = () => {
    setStep(1); setFileData(null); setFile(null); setTrainResult(null)
    setHistory([]); setModelResults({}); setActiveModel(null)
    setTrainError(null); setPrediction(null); setDetailedMetrics(null)
    setXCols([]); setYCol(''); setTab('modelos')
    setPreviewRows([]); setTotalRows(0); setCurrentPage(1)
    setExpandedIds(new Set()); setModelPredictions({})
    activeHistoryId.current = null
  }

  const uploadFile = async () => {
    if (!file) return
    let sid = sessionId
    if (!sid) {
      try {
        const r = await axios.post('/api/session')
        sid = r.data.session_id
        setSessionId(sid)
      } catch (e) {
        alert('No se pudo conectar con el servidor.')
        return
      }
    }
    setUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post('/api/upload?session_id=' + sid, form)
      setFileData(res.data)
      setPreviewRows(res.data.preview)
      setTotalRows(res.data.rows)
      setCurrentPage(1)
      setXCols([]); setYCol(''); setTrainResult(null)
      setPrediction(null); setDetailedMetrics(null); setModelResults({})
      setStep(2)
    } catch (e) {
      alert(e.response?.data?.detail || 'Error al subir el archivo')
    } finally {
      setUploading(false)
    }
  }

  const fetchPreview = async (page, size) => {
    if (!sessionId || !fileData) return
    const offset = (page - 1) * size
    setLoadingPreview(true)
    try {
      const res = await axios.get(`/api/preview/${sessionId}?limit=${size}&offset=${offset}`)
      setPreviewRows(res.data.rows)
      setTotalRows(res.data.total)
    } catch (e) {
      // silent
    }
    setLoadingPreview(false)
  }

  const goToPage = (page) => {
    setCurrentPage(page)
    fetchPreview(page, pageSize)
  }

  const changePageSize = (size) => {
    setPageSize(size)
    setCurrentPage(1)
    fetchPreview(1, size)
  }

  const toggleXCol = col =>
    setXCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])

  const confirmColumns = () => {
    if (!xCols.length || !yCol) return
    setTrainResult(null); setPrediction(null); setDetailedMetrics(null); setStep(3)
  }

  const runModel = async modelKey => {
    if (training) return
    setActiveModel(modelKey); setTraining(true)
    setTrainResult(null); setTrainError(null)
    setPrediction(null); setDetailedMetrics(null); setPredValues({})
    setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
    try {
      const res = await axios.post('/api/train', {
        session_id: sessionId, x_cols: xCols, y_col: yCol, model: modelKey,
      })
      const result = res.data
      const entryId = Date.now()
      activeHistoryId.current = entryId
      setTrainResult(result)
      setModelResults(prev => ({ ...prev, [modelKey]: result }))
      setStep(prev => Math.max(prev, 4))
      setHistory(prev => [{
        id: entryId, modelKey, xCols: [...xCols], yCol,
        model: result.model, type: result.type,
        metrics: result.metrics, summary: result.summary,
        metric: extractMetric(result),
      }, ...prev])
    } catch (e) {
      setTrainError(e.response?.data?.detail || 'Error al entrenar el modelo')
    } finally {
      setTraining(false)
    }
  }

  const runPredict = async () => {
    const values = xCols.map(c => parseFloat(predValues[c] ?? ''))
    if (values.some(isNaN)) { setPredError('Ingresa valores numericos para todas las columnas'); return }
    setPredicting(true); setPredError(null)
    try {
      const res = await axios.post('/api/predict', { session_id: sessionId, values })
      const pred = res.data
      setPrediction(pred)
      const hid = activeHistoryId.current
      if (hid) {
        setModelPredictions(prev => ({
          ...prev,
          [hid]: [...(prev[hid] || []), {
            inputs: xCols.reduce((acc, c, i) => ({ ...acc, [c]: values[i] }), {}),
            result: pred,
            ts: new Date().toLocaleTimeString(),
          }]
        }))
      }
    } catch (e) {
      setPredError(e.response?.data?.detail || 'Error al predecir')
    } finally {
      setPredicting(false)
    }
  }

  const fetchDetailedMetrics = async () => {
    setLoadingMetrics(true)
    try {
      const res = await axios.get('/api/metrics/' + sessionId)
      setDetailedMetrics(res.data)
    } catch (e) {
      alert(e.response?.data?.detail || 'Error al obtener metricas')
    } finally {
      setLoadingMetrics(false)
    }
  }

  const downloadHistory = fmt =>
    window.open('/api/results/' + sessionId + '/download?format=' + fmt, '_blank')

  const mainMetric = extractMetric(trainResult)
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize))

  const NAV_STEPS = [
    { n: 1, label: 'Datos',     Icon: Upload,       done: !!fileData },
    { n: 2, label: 'Columnas',  Icon: Columns,      done: xCols.length > 0 && !!yCol },
    { n: 3, label: 'Modelos',   Icon: BrainCircuit, done: step >= 3 },
    { n: 4, label: 'Resultado', Icon: CheckCircle2, done: !!trainResult },
  ]

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-[#080808] text-gray-900 dark:text-white">

      <header className="flex-shrink-0 h-14 px-6 flex items-center justify-between border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-[#0a0a0a] z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gray-900 dark:bg-white flex items-center justify-center flex-shrink-0">
            <BarChart2 size={14} className="text-white dark:text-black" />
          </div>
          <span className="font-black text-base tracking-tight text-gray-900 dark:text-white">AcgML</span>
        </div>

        <nav className="hidden md:flex items-center">
          {NAV_STEPS.map((s, i) => (
            <div key={s.n} className="flex items-center">
              <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold transition-all
                ${step === s.n ? 'bg-gray-900 text-white dark:bg-white dark:text-black'
                  : s.done ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-gray-400 dark:text-gray-600'}`}>
                {s.done && step !== s.n ? <CheckCircle2 size={12} /> : <s.Icon size={12} />}
                {s.label}
              </div>
              {i < NAV_STEPS.length - 1 && (
                <div className={`w-6 h-px mx-0.5 ${step > s.n ? 'bg-emerald-300 dark:bg-emerald-700' : 'bg-gray-200 dark:bg-gray-800'}`} />
              )}
            </div>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle dark={dark} onToggle={() => setDark(d => !d)} />
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {step >= 3 && (
          <aside className="w-[220px] flex-shrink-0 border-r border-gray-200 dark:border-gray-800 flex flex-col overflow-y-auto bg-white dark:bg-[#0a0a0a]">
            <div className="p-4 border-b border-gray-200 dark:border-gray-800">
              <SectionLabel>Archivo</SectionLabel>
              <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 truncate mb-2">{file?.name}</p>
              <div className="flex gap-1.5 flex-wrap">
                <span className="text-xs px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-[#111] text-gray-600 dark:text-gray-400 font-medium">{fileData?.rows} filas</span>
                <span className="text-xs px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-[#111] text-gray-600 dark:text-gray-400 font-medium">{fileData?.columns?.length} col.</span>
              </div>
            </div>
            <div className="p-4 border-b border-gray-200 dark:border-gray-800">
              <SectionLabel>Variables X</SectionLabel>
              <div className="flex flex-wrap gap-1.5">
                {xCols.map(col => <Tag key={col} label={col} />)}
              </div>
            </div>
            <div className="p-4 border-b border-gray-200 dark:border-gray-800">
              <SectionLabel>Variable Y</SectionLabel>
              {yCol && <Tag label={yCol} />}
            </div>
            {trainResult && (
              <div className="p-4 border-b border-gray-200 dark:border-gray-800">
                <SectionLabel>Ultimo modelo</SectionLabel>
                <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-[#111] p-3">
                  <p className="text-xs font-semibold text-gray-800 dark:text-gray-200 mb-1">{trainResult.model}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500">{trainResult.summary}</p>
                </div>
              </div>
            )}
            <div className="p-4 mt-auto">
              <button onClick={resetToStep1}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#1a1a1a] text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#222] hover:border-gray-300 transition-all">
                <Upload size={12} />
                Cambiar datos
              </button>
            </div>
          </aside>
        )}

        <main className="flex-1 overflow-y-auto flex flex-col">

          {step === 1 && (
            <div className="flex-1 flex items-center justify-center py-16 px-6">
              <div className="w-full max-w-md">
                <div className="text-center mb-10">
                  <div className="w-16 h-16 rounded-2xl bg-gray-900 dark:bg-white flex items-center justify-center mx-auto mb-5 shadow-md">
                    <BarChart2 size={28} className="text-white dark:text-black" />
                  </div>
                  <h1 className="text-3xl font-black text-gray-900 dark:text-white mb-2">AcgML</h1>
                  <p className="text-gray-500 dark:text-gray-500 text-sm leading-relaxed">Carga un archivo CSV para entrenar y comparar modelos de machine learning.</p>
                </div>
                <DropZone file={file} onFile={setFile} />
                {file && (
                  <div className="mt-4">
                    <PrimaryBtn onClick={uploadFile} disabled={uploading} loading={uploading} full>
                      {uploading ? 'Cargando...' : 'Cargar archivo'}
                    </PrimaryBtn>
                  </div>
                )}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="flex-1 flex items-center justify-center py-16 px-6">
              <div className="w-full max-w-2xl">
                <div className="mb-8">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                    <span className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">{file?.name}</span>
                    <span className="text-xs px-2 py-0.5 rounded border border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-500">{fileData?.rows} filas · {fileData?.columns?.length} columnas</span>
                  </div>
                  <h2 className="text-2xl font-black text-gray-900 dark:text-white">Selecciona las variables</h2>
                </div>

                <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden mb-4 shadow-sm">
                  <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-[#111]">
                    <p className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Variables de entrada X</p>
                    <p className="text-xs text-gray-400 dark:text-gray-600 mt-0.5">Una o mas columnas independientes</p>
                  </div>
                  <div className="p-6">
                    <div className="flex flex-wrap gap-2">
                      {fileData?.columns.map(col => {
                        const isNum = fileData.numeric_columns?.includes(col)
                        return isNum
                          ? <ColPill key={col} label={col} selected={xCols.includes(col)} onClick={() => toggleXCol(col)} />
                          : (
                            <span key={col} title="Columna de texto — no usable como entrada X"
                              className="px-3 py-1.5 rounded-xl text-xs font-semibold border border-dashed border-gray-200 dark:border-gray-700 text-gray-300 dark:text-gray-700 cursor-not-allowed line-through">
                              {col}
                            </span>
                          )
                      })}
                    </div>
                    <p className="text-xs text-gray-400 dark:text-gray-600 mt-2">Las columnas tachadas contienen texto y no pueden usarse como X.</p>
                    {xCols.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
                        {xCols.map(col => <Tag key={col} label={col} onRemove={() => toggleXCol(col)} />)}
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden mb-6 shadow-sm">
                  <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-[#111]">
                    <p className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Variable objetivo Y</p>
                    <p className="text-xs text-gray-400 dark:text-gray-600 mt-0.5">La columna que el modelo debe predecir</p>
                  </div>
                  <div className="p-6">
                    <div className="flex flex-wrap gap-2">
                      {fileData?.columns.map(col => (
                        <ColPill key={col} label={col} selected={yCol === col} onClick={() => setYCol(yCol === col ? '' : col)} />
                      ))}
                    </div>
                    {yCol && (
                      <div className="flex gap-1.5 mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
                        <Tag label={yCol} />
                      </div>
                    )}
                  </div>
                </div>

                <PrimaryBtn onClick={confirmColumns} disabled={!xCols.length || !yCol} full>
                  Listo - elegir modelo
                </PrimaryBtn>
              </div>
            </div>
          )}

          {step >= 3 && (
            <div className="flex flex-col flex-1">
              <div className="sticky top-0 z-10 flex items-end px-8 pt-4 bg-gray-50 dark:bg-[#080808] border-b border-gray-200 dark:border-gray-800">
                {[
                  { key: 'modelos',   label: 'Modelos' },
                  { key: 'datos',     label: 'Datos' },
                  { key: 'historial', label: 'Historial', badge: history.length },
                ].map(t => (
                  <button key={t.key} onClick={() => setTab(t.key)}
                    className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold border-b-2 transition-all
                      ${tab === t.key ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-gray-500 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'}`}>
                    {t.label}
                    {t.badge > 0 && (
                      <span className={`text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center
                        ${tab === t.key ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'}`}>
                        {t.badge}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {tab === 'modelos' && (
                <div className="px-8 pb-10 pt-6 flex-1">
                  {training && (
                    <div ref={resultsRef} className="mb-8 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] p-10 flex flex-col items-center gap-5 shadow-sm">
                      <div className="relative w-14 h-14">
                        <div className="absolute inset-0 rounded-full border-4 border-gray-100 dark:border-gray-800" />
                        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-500 animate-spin" />
                        <div className="absolute inset-2 rounded-full bg-gray-50 dark:bg-[#1a1a1a] flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-blue-500" />
                        </div>
                      </div>
                      <div className="text-center">
                        <p className="font-bold text-gray-800 dark:text-gray-200 mb-1">Entrenando modelo</p>
                        <p className="text-sm text-gray-400 dark:text-gray-600">
                          {activeModel === 'mlp_classifier' ? 'Perceptron multicapa — espera un momento...' : 'Calculando...'}
                        </p>
                      </div>
                    </div>
                  )}

                  {trainError && !training && (
                    <div className="mb-8 rounded-2xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-500/10 px-6 py-4">
                      <p className="text-sm text-red-600 dark:text-red-400">{trainError}</p>
                    </div>
                  )}

                  {trainResult && !training && (
                    <div className="mb-8 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden shadow-sm">
                      <div className="px-7 py-5 border-b border-gray-200 dark:border-gray-800 flex flex-wrap items-start justify-between gap-6">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 rounded-full bg-emerald-500" />
                            <span className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                              {trainResult.type === 'regression' ? 'Regresion' : 'Clasificacion'}
                            </span>
                          </div>
                          <h2 className="text-2xl font-black tracking-tight text-gray-900 dark:text-white mb-1">{trainResult.model}</h2>
                          <p className="text-sm text-gray-500 dark:text-gray-400">{xCols.join(' + ')} <span className="text-gray-300 dark:text-gray-700">→</span> {yCol}</p>
                        </div>
                        {mainMetric && <BigNum label={mainMetric.tag} value={mainMetric.display} />}
                      </div>

                      {trainResult.type === 'regression' && (
                        <div className="px-7 py-4 border-b border-gray-200 dark:border-gray-800 grid grid-cols-2 sm:grid-cols-4 gap-3">
                          {trainResult.metrics.r2 !== undefined && <MetricBox label="R2" value={trainResult.metrics.r2} />}
                          {trainResult.metrics.r2_arbol !== undefined && <MetricBox label="R2 Arbol" value={trainResult.metrics.r2_arbol} />}
                          {trainResult.metrics.r2_bosque !== undefined && <MetricBox label="R2 Forest" value={trainResult.metrics.r2_bosque} />}
                          {trainResult.metrics.r2_ridge !== undefined && <MetricBox label="R2 Ridge" value={trainResult.metrics.r2_ridge} />}
                          {trainResult.metrics.r2_lasso !== undefined && <MetricBox label="R2 Lasso" value={trainResult.metrics.r2_lasso} />}
                        </div>
                      )}

                      <div className="px-7 py-5">
                        <SectionLabel>Predecir con nuevos datos</SectionLabel>
                        <div className="flex flex-wrap gap-3 mb-4">
                          {xCols.map(col => (
                            <div key={col}>
                              <label className="block text-xs font-bold text-gray-400 dark:text-gray-600 mb-1 pl-1">{col}</label>
                              <input type="number" step="any" value={predValues[col] ?? ''} onChange={e => setPredValues(p => ({ ...p, [col]: e.target.value }))}
                                className="w-28 rounded-xl border border-gray-200 dark:border-gray-700 px-3 py-2 text-sm text-gray-900 dark:text-white bg-gray-50 dark:bg-[#1a1a1a] focus:outline-none focus:border-blue-400 dark:focus:border-blue-500 transition-colors"
                                placeholder="0" />
                            </div>
                          ))}
                        </div>
                        <div className="flex flex-wrap gap-2 mb-4">
                          <SecondaryBtn onClick={runPredict} disabled={predicting} loading={predicting}>{predicting ? 'Calculando...' : 'Predecir'}</SecondaryBtn>
                          <SecondaryBtn onClick={fetchDetailedMetrics} disabled={loadingMetrics} loading={loadingMetrics}>{loadingMetrics ? 'Cargando...' : 'Metricas detalladas'}</SecondaryBtn>
                        </div>
                        {predError && <p className="text-sm text-red-500 dark:text-red-400 mb-3">{predError}</p>}
                        {prediction && (
                          <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-[#111] p-5 inline-block min-w-[220px]">
                            <p className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-2">Resultado para {yCol}</p>
                            <p className="text-4xl font-black tracking-tighter text-gray-900 dark:text-white mb-2">{prediction.prediction}</p>
                            {prediction.extra && Object.keys(prediction.extra).length > 0 && (
                              <div className="flex gap-4 mb-3">
                                {Object.entries(prediction.extra).map(([k, v]) => (
                                  <div key={k} className="text-xs text-gray-500 dark:text-gray-400">
                                    <span className="capitalize">{k}</span>: <span className="font-semibold text-gray-700 dark:text-gray-300">{v}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                            {prediction.probabilities && (
                              <div className="mt-3 space-y-2.5 min-w-[260px]">
                                {Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a).map(([cls, p]) => <ProbBar key={cls} label={cls} pct={p} />)}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {detailedMetrics && (
                        <div className="border-t border-gray-200 dark:border-gray-800 px-7 py-5">
                          <SectionLabel>Metricas detalladas</SectionLabel>
                          {detailedMetrics.type === 'regression' ? (
                            <div className="grid grid-cols-3 gap-3 max-w-md">
                              <MetricBox label="R2" value={detailedMetrics.r2} />
                              <MetricBox label="MAE" value={detailedMetrics.mae} />
                              <MetricBox label="MSE" value={detailedMetrics.mse} />
                            </div>
                          ) : (
                            <div className="overflow-x-auto">
                              <table className="text-sm border-collapse">
                                <thead>
                                  <tr>{['Clase','Precision','Recall','F1-Score','Soporte'].map(h => (
                                    <th key={h} className="text-left py-2 pr-8 text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 border-b border-gray-200 dark:border-gray-800">{h}</th>
                                  ))}</tr>
                                </thead>
                                <tbody>
                                  {Object.entries(detailedMetrics.report).filter(([k]) => !['accuracy','macro avg','weighted avg'].includes(k)).map(([cls, v]) => (
                                    <tr key={cls} className="border-b border-gray-100 dark:border-gray-800">
                                      <td className="py-2.5 pr-8 font-semibold text-gray-800 dark:text-gray-200">{cls}</td>
                                      <td className="py-2.5 pr-8 text-gray-600 dark:text-gray-400">{(v.precision*100).toFixed(1)}%</td>
                                      <td className="py-2.5 pr-8 text-gray-600 dark:text-gray-400">{(v.recall*100).toFixed(1)}%</td>
                                      <td className="py-2.5 pr-8 text-gray-600 dark:text-gray-400">{(v['f1-score']*100).toFixed(1)}%</td>
                                      <td className="py-2.5 pr-8 text-gray-600 dark:text-gray-400">{v.support}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="mb-8">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-2 h-2 rounded-full bg-blue-500" />
                      <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Regresion</p>
                      <div className="flex-1 h-px bg-gray-200 dark:bg-gray-800" />
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
                      {REGRESSION_MODELS.map(m => (
                        <ModelCard key={m.key} model={m} color="bg-blue-500" active={activeModel === m.key} loading={training} result={modelResults[m.key]} onClick={() => runModel(m.key)} />
                      ))}
                    </div>
                  </div>

                  <div className="mb-8">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-2 h-2 rounded-full bg-violet-500" />
                      <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Clasificacion</p>
                      <div className="flex-1 h-px bg-gray-200 dark:bg-gray-800" />
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                      {CLASSIFICATION_MODELS.map(m => (
                        <ModelCard key={m.key} model={m} color='bg-violet-500' active={activeModel === m.key} loading={training} result={modelResults[m.key]} onClick={() => runModel(m.key)} />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {tab === 'datos' && (
                <div className="px-8 pb-10 pt-6 flex-1">
                  <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
                    <div>
                      <h2 className="text-xl font-black text-gray-900 dark:text-white">Vista previa del dataset</h2>
                      <p className="text-sm text-gray-500 dark:text-gray-500 mt-0.5">
                        {totalRows} filas totales · {fileData?.columns?.length} columnas
                        {loadingPreview && <span className="ml-2 text-blue-500 text-xs">cargando...</span>}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs text-gray-400 dark:text-gray-600 font-medium">Filas por pagina:</span>
                      {PAGE_SIZES.map(s => (
                        <button key={s} onClick={() => changePageSize(s)} disabled={loadingPreview}
                          className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all
                            ${pageSize === s
                              ? 'bg-gray-900 text-white border-gray-900 dark:bg-white dark:text-black dark:border-white'
                              : 'border-gray-200 text-gray-500 hover:border-gray-400 hover:text-gray-700 dark:border-gray-700 dark:text-gray-400 dark:hover:border-gray-500 disabled:opacity-40'}`}>
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden shadow-sm">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse">
                        <thead>
                          <tr className="bg-gray-50 dark:bg-[#111] border-b border-gray-200 dark:border-gray-800">
                            <th className="text-left px-4 py-3 text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 w-10">#</th>
                            {fileData?.columns?.map(col => (
                              <th key={col} className="text-left px-4 py-3 text-[11px] font-bold uppercase tracking-widest whitespace-nowrap">
                                <span className={xCols.includes(col) ? 'text-blue-600 dark:text-blue-400' : col === yCol ? 'text-emerald-700 dark:text-emerald-400' : 'text-gray-400 dark:text-gray-600'}>
                                  {col}
                                  {xCols.includes(col) && <span className="ml-1 text-[9px] normal-case opacity-60">X</span>}
                                  {col === yCol && <span className="ml-1 text-[9px] normal-case opacity-60">Y</span>}
                                </span>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {previewRows.map((row, i) => {
                            const rowNum = (currentPage - 1) * pageSize + i + 1
                            return (
                              <tr key={i} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors">
                                <td className="px-4 py-3 text-xs text-gray-400 dark:text-gray-600 font-mono tabular-nums">{rowNum}</td>
                                {fileData?.columns?.map(col => (
                                  <td key={col} className={`px-4 py-3 tabular-nums whitespace-nowrap
                                    ${xCols.includes(col) ? 'text-gray-800 dark:text-gray-200 font-medium' : col === yCol ? 'text-emerald-700 dark:text-emerald-400 font-bold' : 'text-gray-500 dark:text-gray-500'}`}>
                                    {typeof row[col] === 'number' ? (Number.isInteger(row[col]) ? row[col] : row[col].toFixed(4)) : String(row[col])}
                                  </td>
                                ))}
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>

                    <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-[#111] flex items-center justify-between flex-wrap gap-3">
                      <p className="text-xs text-gray-400 dark:text-gray-600">
                        Filas {(currentPage - 1) * pageSize + 1}–{Math.min(currentPage * pageSize, totalRows)} de {totalRows}
                      </p>
                      <div className="flex items-center gap-1">
                        <button onClick={() => goToPage(1)} disabled={currentPage === 1 || loadingPreview}
                          className="px-2 py-1.5 rounded-lg text-xs font-bold border border-gray-200 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors">
                          «
                        </button>
                        <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1 || loadingPreview}
                          className="px-2.5 py-1.5 rounded-lg text-xs font-bold border border-gray-200 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors">
                          ‹
                        </button>
                        {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                          let p
                          if (totalPages <= 5) p = i + 1
                          else if (currentPage <= 3) p = i + 1
                          else if (currentPage >= totalPages - 2) p = totalPages - 4 + i
                          else p = currentPage - 2 + i
                          return (
                            <button key={p} onClick={() => goToPage(p)} disabled={loadingPreview}
                              className={`w-8 h-7 rounded-lg text-xs font-bold border transition-all
                                ${currentPage === p
                                  ? 'bg-gray-900 text-white border-gray-900 dark:bg-white dark:text-black dark:border-white'
                                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-white/5'}`}>
                              {p}
                            </button>
                          )
                        })}
                        <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage === totalPages || loadingPreview}
                          className="px-2.5 py-1.5 rounded-lg text-xs font-bold border border-gray-200 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors">
                          ›
                        </button>
                        <button onClick={() => goToPage(totalPages)} disabled={currentPage === totalPages || loadingPreview}
                          className="px-2 py-1.5 rounded-lg text-xs font-bold border border-gray-200 dark:border-gray-700 disabled:opacity-30 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors">
                          »
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {tab === 'historial' && (
                <div className="px-8 pb-10 pt-6 flex-1">
                  {history.length === 0 ? (
                    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] flex flex-col items-center justify-center py-24">
                      <BarChart2 size={28} className="text-gray-300 dark:text-gray-700 mb-4" />
                      <p className="text-sm text-gray-500 dark:text-gray-500">Entrena modelos para ver el historial.</p>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-center justify-between mb-6">
                        <div>
                          <h2 className="text-xl font-black text-gray-900 dark:text-white">Historial de entrenamiento</h2>
                          <p className="text-sm text-gray-500 dark:text-gray-500 mt-0.5">{history.length} modelo{history.length > 1 ? 's' : ''} — clic en cada fila para ver detalles</p>
                        </div>
                        <div className="flex gap-2">
                          <SecondaryBtn onClick={() => downloadHistory('json')}>Exportar JSON</SecondaryBtn>
                          <SecondaryBtn onClick={() => downloadHistory('csv')}>Exportar CSV</SecondaryBtn>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden shadow-sm">
                        <div className="grid grid-cols-[28px_1fr_auto_160px] gap-4 items-center px-6 py-3 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-[#111]">
                          {['#','Modelo','Resultado','Metrica'].map(h => (
                            <span key={h} className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">{h}</span>
                          ))}
                        </div>

                        {history.map((entry, i) => {
                          const isReg = entry.type === 'regression'
                          const isExpanded = expandedIds.has(entry.id)
                          const preds = modelPredictions[entry.id] || []
                          return (
                            <div key={entry.id} className="border-b last:border-0 border-gray-100 dark:border-gray-800">
                              <button
                                onClick={() => setExpandedIds(prev => { const s = new Set(prev); s.has(entry.id) ? s.delete(entry.id) : s.add(entry.id); return s })}
                                className="w-full grid grid-cols-[28px_1fr_auto_160px] gap-4 items-center px-6 py-4 hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors text-left">
                                <span className="text-sm font-bold text-gray-400 dark:text-gray-600 tabular-nums">{history.length - i}</span>
                                <div className="min-w-0">
                                  <div className="flex items-center gap-2 mb-0.5">
                                    {isExpanded ? <ChevronDown size={13} className="text-gray-400 flex-shrink-0" /> : <ChevronRight size={13} className="text-gray-400 flex-shrink-0" />}
                                    <span className="font-semibold text-sm text-gray-800 dark:text-gray-200 truncate">{entry.model}</span>
                                  </div>
                                  <div className="flex items-center gap-2 pl-5">
                                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold border ${isReg ? 'bg-blue-50 text-blue-600 border-blue-100 dark:bg-blue-500/10 dark:text-blue-400 dark:border-blue-500/20' : 'bg-violet-50 text-violet-600 border-violet-100 dark:bg-violet-500/10 dark:text-violet-400 dark:border-violet-500/20'}`}>
                                      {isReg ? 'Regresion' : 'Clasificacion'}
                                    </span>
                                    <span className="text-xs text-gray-400 dark:text-gray-600 truncate">{entry.xCols.join(', ')} → {entry.yCol}</span>
                                    {preds.length > 0 && (
                                      <span className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-gray-100 dark:bg-white/5 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                                        {preds.length} prediccion{preds.length > 1 ? 'es' : ''}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                <div className="text-right">
                                  {entry.metric && (
                                    <>
                                      <p className="text-lg font-black text-gray-900 dark:text-white tabular-nums leading-tight">{entry.metric.display}</p>
                                      <p className="text-[11px] text-gray-400 dark:text-gray-600">{entry.metric.tag}</p>
                                    </>
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  <div className="flex-1 h-2 rounded-full bg-gray-100 dark:bg-white/10 overflow-hidden">
                                    <div className={`h-full rounded-full ${isReg ? 'bg-blue-500' : 'bg-violet-500'}`}
                                      style={{ width: entry.metric ? Math.min(100, Math.max(0, Math.round(entry.metric.raw * 100))) + '%' : '0%' }} />
                                  </div>
                                  <span className="text-xs font-bold text-gray-500 dark:text-gray-400 w-8 text-right tabular-nums">
                                    {entry.metric ? Math.round(entry.metric.raw * 100) + '%' : '—'}
                                  </span>
                                </div>
                              </button>

                              {isExpanded && (
                                <div className="px-6 pb-6 pt-2 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-[#0f0f0f] space-y-5">
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3">
                                    <div className="sm:col-span-2">
                                      <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3">Metricas de entrenamiento</p>
                                      <div className="flex flex-wrap gap-3">
                                        {Object.entries(entry.metrics).map(([k, v]) => (
                                          <div key={k} className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] px-4 py-3 min-w-[100px]">
                                            <p className="text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1">{k.replace(/_/g, ' ')}</p>
                                            <p className="text-xl font-black text-gray-900 dark:text-white tabular-nums">
                                              {k === 'accuracy' ? v.toFixed(2) + '%' : typeof v === 'number' ? v.toFixed(4) : v}
                                            </p>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                    <div>
                                      <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3">Configuracion</p>
                                      <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] p-4 space-y-3">
                                        <div>
                                          <p className="text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1.5">Entradas X</p>
                                          <div className="flex flex-wrap gap-1">
                                            {entry.xCols.map(c => <span key={c} className="text-xs px-2 py-0.5 rounded-lg bg-blue-50 text-blue-700 border border-blue-100 dark:bg-blue-500/10 dark:text-blue-300 dark:border-blue-500/20 font-semibold">{c}</span>)}
                                          </div>
                                        </div>
                                        <div>
                                          <p className="text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1.5">Objetivo Y</p>
                                          <span className="text-xs px-2 py-0.5 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-100 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/20 font-semibold">{entry.yCol}</span>
                                        </div>
                                        <div>
                                          <p className="text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1">Resumen</p>
                                          <p className="text-xs text-gray-600 dark:text-gray-400">{entry.summary}</p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  {preds.length > 0 && (
                                    <div>
                                      <p className="text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3">Predicciones realizadas ({preds.length})</p>
                                      <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-[#141414] overflow-hidden">
                                        <table className="w-full text-sm border-collapse">
                                          <thead>
                                            <tr className="bg-gray-50 dark:bg-[#111] border-b border-gray-200 dark:border-gray-800">
                                              <th className="text-left px-4 py-2.5 text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">#</th>
                                              {entry.xCols.map(c => (
                                                <th key={c} className="text-left px-4 py-2.5 text-[11px] font-bold uppercase tracking-widest text-blue-500 dark:text-blue-400">{c}</th>
                                              ))}
                                              <th className="text-left px-4 py-2.5 text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400">{entry.yCol}</th>
                                              {isReg || <th className="text-left px-4 py-2.5 text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Probabilidades</th>}
                                              <th className="text-left px-4 py-2.5 text-[11px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-600">Hora</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {preds.map((p, pi) => (
                                              <tr key={pi} className="border-b last:border-0 border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-white/[0.02]">
                                                <td className="px-4 py-2.5 text-xs text-gray-400 dark:text-gray-600 font-mono">{pi + 1}</td>
                                                {entry.xCols.map(c => (
                                                  <td key={c} className="px-4 py-2.5 text-sm text-gray-600 dark:text-gray-400 tabular-nums">{p.inputs[c]}</td>
                                                ))}
                                                <td className="px-4 py-2.5 text-sm font-bold text-emerald-700 dark:text-emerald-400 tabular-nums">{p.result.prediction}</td>
                                                {!isReg && (
                                                  <td className="px-4 py-2.5">
                                                    {p.result.probabilities ? (
                                                      <div className="flex flex-wrap gap-1.5">
                                                        {Object.entries(p.result.probabilities).sort(([,a],[,b]) => b-a).slice(0,3).map(([cls, prob]) => (
                                                          <span key={cls} className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-gray-100 dark:bg-white/5 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                                                            {cls}: {(prob*100).toFixed(1)}%
                                                          </span>
                                                        ))}
                                                      </div>
                                                    ) : '—'}
                                                  </td>
                                                )}
                                                <td className="px-4 py-2.5 text-xs text-gray-400 dark:text-gray-600 font-mono">{p.ts}</td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                      </div>
                                    </div>
                                  )}

                                  {preds.length === 0 && (
                                    <p className="text-xs text-gray-400 dark:text-gray-600 italic">No se realizaron predicciones con este modelo.</p>
                                  )}
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          <footer className="flex-shrink-0 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-[#0a0a0a] px-8 py-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
              <div>
                <p className="text-xs font-semibold text-gray-700 dark:text-gray-300">L.I.A. Armando Cruz Guerrero</p>
                <p className="text-xs text-gray-400 dark:text-gray-600 mt-0.5">Inteligencia Artificial · Maestria en Ciencias de la Computacion</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-400 dark:text-gray-600">En colaboracion con</p>
                <p className="text-xs font-semibold text-gray-600 dark:text-gray-400">Dr. Rafael Rojas Hernandez</p>
              </div>
            </div>
          </footer>
        </main>
      </div>
    </div>
  )
}

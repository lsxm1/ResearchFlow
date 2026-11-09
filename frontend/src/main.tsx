import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Paper = {id: string; title: string; filename: string; status: string; page_count: number; processed_pages: number; text_pages: number; file_url: string; error?: string; version?: number};
type Evidence = {id: string; document_id: string; title: string; filename: string; page: number; text: string; bbox: number[]; score: number; file_url: string; source?: 'native' | 'ocr'};
type Task = {id: string; question: string; document_ids: string[]; status: string; events: {step: string; message: string; tool: string; time: string}[]; result: null | {mode: string; claims: {text: string; source_id: string; quote: string}[]; evidence: Evidence[]; warnings: string[]}};
type Health = {model_mode: string; model_name?: string; version: string; ocr_available: boolean};
type Duplicate = {document_id: string; title: string; similarity: number; recommendation: string};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `请求失败 (${response.status})`);
  }
  return response.status === 204 ? undefined as T : response.json();
}

const statuses: Record<string, string> = {queued: '等待解析', parsing: '解析中', ready: '可检索', partial: '部分页面需OCR', needs_ocr: '需要OCR', failed: '失败'};
const roadmap = [
  ['01', '文献入库与解析', '已实现基础版', '批量上传、逐页进度、英文扫描件OCR、正文相似提醒。'],
  ['02', '证据问答与对比', '已实现基础版', 'BM25检索、按篇证据对照、页码与原文块。'],
  ['03', '联动阅读与翻译', '待开发', '当前支持原始PDF与提取文本切换。后续增加标题导航和段落对照翻译。'],
  ['04', '论文搜索与追踪', '待开发', '按标题搜索、预览后导入、主题追踪和可解释推荐。'],
  ['05', '引用图谱与综述', '待开发', '从真实参考文献建立局域图谱、PageRank和带证据的综述。'],
  ['06', '写作与图表', '待开发', '从证据生成提纲和参考文献；CSV生成图表、脚本和图注。'],
];

function EvidenceCards({sources, openPaper}: {sources: Evidence[]; openPaper: (id: string, page: number) => void}) {
  return <>{sources.map((source, index) => <article className="evidence" key={source.id}>
    <div className="evidence-head">
      <span className="source-number">{String(index + 1).padStart(2, '0')}</span>
      <button onClick={() => openPaper(source.document_id, source.page)}>第 {source.page} 页 ↗</button>
    </div>
    <strong>{source.title}</strong>
    <p>{source.text}</p>
    <small>BM25 {source.score.toFixed(2)} · {source.source === 'ocr' ? 'OCR 识别片段' : '原文块已定位'}</small>
  </article>)}</>;
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [current, setCurrent] = useState<string>('');
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState('1');
  const [question, setQuestion] = useState('What hardware and training schedule were used?');
  const [health, setHealth] = useState<Health | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [view, setView] = useState<'workspace' | 'roadmap'>('workspace');
  const [readerMode, setReaderMode] = useState<'pdf' | 'text'>('pdf');
  const [pageText, setPageText] = useState('');
  const [pageFromOcr, setPageFromOcr] = useState(false);
  const [duplicates, setDuplicates] = useState<Duplicate[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const paper = papers.find(x => x.id === current);
  const task = tasks.find(x => x.id === taskId);

  useEffect(() => {
    let active = true;
    let busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      try {
        const [docs, h, history] = await Promise.all([api<Paper[]>('/documents'), api<Health>('/health'), api<Task[]>('/tasks')]);
        if (active) {setPapers(docs); setHealth(h); setTasks(history);}
      } catch { if (active) setHealth(null); }
      finally { busy = false; }
    };
    void refresh();
    const timer = setInterval(refresh, 1800);
    return () => {active = false; clearInterval(timer);};
  }, []);

  useEffect(() => {
    let active = true;
    setDuplicates([]);
    if (current && paper && !['queued', 'parsing'].includes(paper.status)) {
      api<Paper & {duplicates: Duplicate[]}>(`/documents/${current}`).then(d => {if (active) setDuplicates(d.duplicates);}).catch(() => {});
    }
    return () => {active = false;};
  }, [current, paper?.status, papers.length]);

  useEffect(() => {
    let active = true;
    setPageText('');
    setPageFromOcr(false);
    if (current && readerMode === 'text') {
      api<{text: string; source: string}>(`/documents/${current}/pages/${page}`).then(p => {
        if (active) {
          setPageText(p.text || '本页没有可提取文本，需要OCR或本页为空白页。');
          setPageFromOcr(p.source === 'ocr');
        }
      }).catch(e => {if (active) setPageText(e.message);});
    }
    return () => {active = false;};
  }, [current, page, readerMode, paper?.processed_pages, paper?.status]);

  function openPaper(id: string, targetPage = 1) {
    setCurrent(id); setPage(targetPage); setPageInput(String(targetPage)); setView('workspace');
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true); setError(''); setNotice('');
    let completed = 0;
    try {
      for (const file of Array.from(files)) {
        const form = new FormData(); form.append('file', file);
        const response = await api<{document: Paper; exact_duplicate: boolean}>('/documents', {method: 'POST', body: form});
        openPaper(response.document.id);
        completed++;
        if (response.exact_duplicate) setNotice('相同文件已存在，已打开原有文献。');
      }
      setPapers(await api<Paper[]>('/documents'));
      setNotice(`${completed}份文件已处理，解析在后台继续。`);
    } catch (e) {setError(`${completed}份文件已处理。${(e as Error).message}`);}
    finally {setUploading(false); if (fileInput.current) fileInput.current.value = '';}
  }

  async function submit() {
    setError(''); setSubmitting(true);
    try {
      const valid = selected.filter(id => papers.some(p => p.id === id && ['ready','partial'].includes(p.status)));
      const response = await api<{id: string}>('/tasks', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question, document_ids: valid})});
      setTaskId(response.id);
      setTasks(await api<Task[]>('/tasks'));
    } catch(e) {setError((e as Error).message);}
    finally {setSubmitting(false);}
  }

  async function remove(id: string) {
    try {
      await api(`/documents/${id}`, {method: 'DELETE'});
      setPapers(p => p.filter(x => x.id !== id)); setSelected(p => p.filter(x => x !== id));
      if (current === id) setCurrent('');
    } catch(e) {setError((e as Error).message);}
  }

  async function retryOcr(id: string) {
    setError('');
    try {
      await api(`/documents/${id}/retry-ocr`, {method: 'POST'});
      setPapers(await api<Paper[]>('/documents'));
    } catch (e) {setError((e as Error).message);}
  }

  function jump(value: number) {
    const next = Math.min(paper?.page_count || 1, Math.max(1, value || 1));
    setPage(next); setPageInput(String(next));
  }

  const eligibleCount = selected.filter(id => papers.some(p => p.id === id && ['ready','partial'].includes(p.status))).length;
  return <div className="app">
    <aside className="nav">
      <div className="brand"><span className="brand-symbol">R<span>·</span></span><div>ResearchFlow<small>科研证据工作台</small></div></div>
      <div className="nav-label">研究空间</div>
      <button className={view === 'workspace' ? 'nav-item active' : 'nav-item'} onClick={() => setView('workspace')}><span>▤</span>文献与任务</button>
      <button className={view === 'roadmap' ? 'nav-item active' : 'nav-item'} onClick={() => setView('roadmap')}><span>▦</span>模块计划</button>
      <div className="nav-bottom"><span className="version">LOCAL WORKSPACE / V0.1</span><p>从论文出发<br/>让每一次判断有据可查。</p><span className={health ? 'connection online' : 'connection'}>● {health ? '本地服务已连接' : '等待后端服务'}</span></div>
    </aside>
    <main>
      <header><div className="breadcrumb">我的研究 / <strong>{view === 'workspace' ? '文献与任务' : '开发路线'}</strong></div><span className="mode-badge">{health?.model_mode === 'configured' ? `模型已配置 · ${health.model_name}` : '证据检索模式 · 无模型调用'}</span></header>
      <div className="page-heading"><div><div className="eyebrow">RESEARCH, GROUNDED.</div><h1>{view === 'workspace' ? '把论文，变成可追溯的证据。' : '六个模块，一个科研闭环。'}</h1><p>导入文献，寻找证据，回到原文。先验证，再形成结论。</p></div><span className="phase">第一阶段<span>文献 → 证据</span></span></div>
      {error && <div className="banner error" role="alert">{error}<button onClick={() => setError('')} aria-label="关闭错误提示">×</button></div>}
      {notice && <div className="banner" role="status">{notice}<button onClick={() => setNotice('')} aria-label="关闭提示">×</button></div>}
      {!health && <div className="banner error">后端尚未连接。请运行项目中的 start-backend.ps1。</div>}
      {view === 'roadmap' ? <section className="roadmap">{roadmap.map(([num, title, state, desc]) => <article className="module-card" key={num}><span className="module-number">{num}</span><span className={`state ${state.includes('已实现') ? 'ready' : ''}`}>{state}</span><h2>{title}</h2><p>{desc}</p></article>)}</section> : <div className="workspace">
        <section className="panel library"><div className="panel-heading"><div><h2>文献库 <span className="count">{papers.length}</span></h2><p>勾选文献，限定任务证据范围</p></div></div>
          <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple aria-label="上传PDF文件" onChange={e => void upload(e.target.files)} hidden/>
          <button className="upload" disabled={uploading || !health} onClick={() => fileInput.current?.click()}><span>＋</span>{uploading ? '上传中…' : '导入 PDF 文献'}<small>可多选 · 每份最多50MB</small></button>
          <div className="paper-list">{!papers.length ? <div className="empty small"><span>▧</span><h3>研究从一篇论文开始</h3><p>上传你的PDF，解析进度会显示在这里。</p></div> : papers.map(p => <article className={`paper ${current === p.id ? 'selected' : ''}`} key={p.id}>
            <div className="paper-top"><input type="checkbox" aria-label={`将${p.filename}加入任务`} checked={selected.includes(p.id)} disabled={!['ready','partial'].includes(p.status)} onChange={e => setSelected(s => e.target.checked ? [...s, p.id] : s.filter(x => x !== p.id))}/><button className="paper-title" onClick={() => openPaper(p.id)} title={p.title}>{p.title}</button></div>
            <div className="filename" title={p.filename}>{p.filename}</div><div className="paper-meta"><span className={`state ${p.status}`}>{statuses[p.status] || p.status}</span><span>{p.page_count ? `${p.processed_pages}/${p.page_count}页` : '排队中'}</span></div>
            {['queued','parsing'].includes(p.status) && <progress max={p.page_count || 1} value={p.processed_pages} aria-label="解析进度"/>}
            {p.error && <p className="inline-error">{p.error}</p>}
            <div className="paper-actions"><button onClick={() => openPaper(p.id)}>打开原文 ↗</button><button onClick={() => void remove(p.id)} aria-label={`删除${p.filename}`}>移除</button></div>
          </article>)}</div>
          <div className="library-footer">{eligibleCount} 篇已选 · 当前为单用户本地工作区</div>
        </section>
        <section className="panel reader"><div className="panel-heading"><div><h2>原文阅读</h2><p>{paper ? paper.filename : '查看证据对应的PDF页面'}</p></div><div className="segmented"><button className={readerMode === 'pdf' ? 'chosen' : ''} onClick={() => setReaderMode('pdf')}>PDF</button><button className={readerMode === 'text' ? 'chosen' : ''} onClick={() => setReaderMode('text')}>提取文本</button></div></div>
          {paper ? <>
            <div className="reader-toolbar"><button onClick={() => jump(page-1)} disabled={page <= 1} aria-label="上一页">←</button><form onSubmit={e => {e.preventDefault(); jump(Number(pageInput));}}><input type="number" min="1" max={paper.page_count || 1} value={pageInput} onChange={e => setPageInput(e.target.value)} aria-label="PDF页码"/> / {paper.page_count || '…'} 页 <button type="submit">跳转</button></form><button onClick={() => jump(page+1)} disabled={page >= paper.page_count} aria-label="下一页">→</button><a href={`${paper.file_url}#page=${page}`} target="_blank" rel="noreferrer">新窗口 ↗</a></div>
            {duplicates.map(d => <div className="duplicate" key={d.document_id}><strong>发现正文相似文献 · {Math.round(d.similarity * 100)}%</strong><span>{d.recommendation}</span><button onClick={() => openPaper(d.document_id)}>查看已有文献 ↗</button></div>)}
            {['needs_ocr', 'partial'].includes(paper.status) && <div className="reader-note">
              {health?.ocr_available
                ? <>仍有页面需要识别。 <button onClick={() => void retryOcr(paper.id)}>重新进行 OCR ↗</button></>
                : '仍有页面需要OCR。运行 setup-ocr.ps1 准备本地英文识别数据。'}
            </div>}
            {readerMode === 'text' && pageFromOcr && <div className="reader-note">本页文本来自 OCR，关键数字请对照 PDF 核实。</div>}
            {readerMode === 'pdf' ? <iframe key={`${current}:${page}`} src={`${paper.file_url}#page=${page}&view=FitH`} title={`${paper.filename} 第${page}页`}/> : <pre className="page-text">{pageText || '正在读取本页…'}</pre>}
          </> : <div className="empty reader-empty"><div className="page-illustration"><span/><span/><span/><span/><span/></div><h3>原文始终在手边</h3><p>打开一篇文献，或点击检索结果中的出处。<br/>每条证据都保留论文、页码和原文片段。</p></div>}
        </section>
        <section className="panel agent"><div className="panel-heading"><div><h2>研究任务</h2><p>限定证据范围，让结果可核查</p></div><span className="tiny-mark">RF</span></div>
          <div className="task-compose"><label htmlFor="question">你希望从论文中找到什么？</label><textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} maxLength={2000}/><div className="examples"><button onClick={() => setQuestion('What hardware and training schedule were used?')}>训练硬件</button><button onClick={() => setQuestion('What are the Adam beta and learning rate hyperparameters?')}>超参数</button><button onClick={() => setQuestion('Compare the BLEU results for English to German and English to French.')}>实验指标</button></div><button className="primary" onClick={() => void submit()} disabled={!health || !eligibleCount || question.trim().length < 2 || submitting}>{submitting ? '提交中…' : health?.model_mode === 'configured' ? '执行研究任务 →' : '检索原文证据 →'}</button><p className="compose-note">{health?.model_mode === 'configured' ? '使用后端配置的模型；引用校验通过仍需人工核对。' : '当前不生成AI答案，不消耗大模型API额度。'}</p></div>
          {tasks.length > 0 && <div className="task-history"><label htmlFor="history">任务记录</label><select id="history" value={taskId} onChange={e => setTaskId(e.target.value)}><option value="">选择已保存的任务</option>{tasks.map(t => <option key={t.id} value={t.id}>{t.question.slice(0,55)} · {t.status}</option>)}</select></div>}
          <div className="task-output" aria-live="polite">{task ? <>
            <div className="output-title"><h3>执行记录</h3><span className={`state ${task.status}`}>{({queued:'排队中',running:'执行中',completed:'已完成',failed:'失败'} as Record<string,string>)[task.status]}</span></div>
            <ol className="timeline">{task.events.map((event,i) => <li key={i}><strong>{event.tool}</strong><span>{event.message}</span></li>)}</ol>
            {task.result?.warnings.map((w,i) => <p className="warning" key={i}>{w}</p>)}
            {!!task.result?.claims.length && <><h3>有来源的结论</h3>{task.result.claims.map((claim,i) => {const source = task.result?.evidence.find(x => x.id === claim.source_id); return <article className="claim" key={i}><p>{claim.text}</p><blockquote>{claim.quote}</blockquote>{source && <button onClick={() => openPaper(source.document_id,source.page)}>原文 · 第{source.page}页 ↗</button>}</article>;})}</>}
            {task.result && task.result.mode !== 'error' && task.document_ids.length > 1 ? <>
              <div className="output-title"><h3>多文献证据对照</h3><span>{task.document_ids.length} 篇</span></div>
              <p className="result-note">各篇按关键词相关性排序。点击页码核对片段。</p>
              {task.document_ids.map(id => {
                const sources = task.result?.evidence.filter(source => source.document_id === id) ?? [];
                const filename = papers.find(p => p.id === id)?.filename ?? sources[0]?.filename ?? '已移除文献';
                return <section key={id}>
                  <div className="output-title"><h3>{filename}</h3><span>{sources.length} 条</span></div>
                  {sources.length ? <EvidenceCards sources={sources} openPaper={openPaper}/> : <p className="result-note">这篇文献未返回匹配片段。</p>}
                </section>;
              })}
            </> : !!task.result?.evidence.length && <>
              <div className="output-title"><h3>原文候选</h3><span>{task.result.evidence.length} 条</span></div>
              <p className="result-note">关键词相关性排序。点击页码核对片段。</p>
              <EvidenceCards sources={task.result.evidence} openPaper={openPaper}/>
            </>}
          </> : <div className="empty small task-empty"><span>◎</span><h3>让证据先行</h3><p>在左侧勾选已解析文献，再发起任务。<br/>执行步骤与来源将保存在这里。</p></div>}</div>
        </section>
      </div>}
      <footer>ResearchFlow / 从原文到证据，再到判断。<span>开发用公开替代数据集 · 不代表官方验收结果</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);

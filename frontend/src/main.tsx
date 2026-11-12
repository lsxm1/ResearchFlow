import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { api, post } from './api';
import { Discovery, Graph, Reading, Writing } from './ResearchTools';
import './style.css';

type Paper = {id: string; title: string; filename: string; status: string; page_count: number; processed_pages: number; text_pages: number; file_url: string; error?: string; version?: number; archived: boolean};
type Evidence = {id: string; document_id: string; title: string; filename: string; page: number; text: string; bbox: number[]; score: number; file_url: string; source?: 'native' | 'ocr'; table?: {caption: string; row: number; column: number; heading: string; model?: string}};
type Citation = {source_id: string; quote: string};
type ComparisonExcerpt = {label: string; display?: string; source_id?: string; quote?: string; searched_queries: string[];
  alternatives?: {label: string; display: string; source_id: string; quote: string}[]};
type ComparisonCell = {document_id: string; status: 'missing' | 'candidate' | 'cited_summary' | 'verified_excerpt' | 'partial' | 'conflict'; source_id?: string; quote?: string; text?: string; searched_queries?: string[]; excerpts?: ComparisonExcerpt[]};
type Comparison = {
  papers: {id: string; title: string; filename: string}[];
  rows: {key: string; label: string; cells: ComparisonCell[]}[];
  findings?: {label: string; text: string; citations: Citation[]}[];
};
type Task = {id: string; question: string; document_ids: string[]; status: string; events: {step: string; message: string; tool: string; time: string}[]; result: null | {mode: string; claims: {text: string; source_id: string; quote: string}[]; evidence: Evidence[]; warnings: string[]; comparison?: Comparison}};
type Health = {model_mode: string; model_name?: string; version: string; ocr_available: boolean};
type Duplicate = {document_id: string; title: string; filename: string; version?: number; similarity: number; recommendation: string};

const statuses: Record<string, string> = {queued: '等待解析', parsing: '解析中', ready: '可检索', partial: '部分页面需OCR', needs_ocr: '需要OCR', failed: '失败'};
const views = {
  workspace: '文献与任务', reading: '联动阅读', discovery: '检索与追踪',
  graph: '引用图谱', writing: '写作与绘图', roadmap: '模块计划',
} as const;
type View = keyof typeof views;
const roadmap = [
  ['01', '文献入库与解析', '基础版', '批量上传、OCR、正文相似提醒、目录与页面结构候选。'],
  ['02', '证据问答与对比', '基础版', '按问题选对比字段、补查证据缺口、原文溯源。模型摘要需配置服务。'],
  ['03', '联动阅读与翻译', '基础版', '原版预览、目录和引用跳转。划选与整篇翻译需配置模型。'],
  ['04', '论文搜索与追踪', '基础版', 'arXiv 搜索导入、定期刷新主题、推荐理由和反馈。'],
  ['05', '引用图谱与综述', '基础版', '引用证据、局域 PageRank、简述和未来工作线索。'],
  ['06', '写作与图表', '基础版', '带引用提纲、CSV 图表和图注、Mermaid 与 Matplotlib 脚本。'],
];

function EvidenceCards({sources, openEvidence}: {sources: Evidence[]; openEvidence: (source: Evidence) => void}) {
  return <>{sources.map((source, index) => <article className="evidence" key={source.id}>
    <div className="evidence-head">
      <span className="source-number">{String(index + 1).padStart(2, '0')}</span>
      <button onClick={() => openEvidence(source)}>第 {source.page} 页 ↗</button>
    </div>
    <strong>{source.title}</strong>
    <p>{source.text}</p>
    <small>BM25 {source.score.toFixed(2)} · {source.source === 'ocr' ? 'OCR 识别片段' : '原文块已定位'}</small>
  </article>)}</>;
}

function SourceExcerpt({source, quote, openEvidence, title}: {source: Evidence; quote?: string; openEvidence: (source: Evidence) => void; title?: string}) {
  const table = source.table;
  return <div className="source-excerpt">
    <button className="source-jump" onClick={() => openEvidence(source)}>
      {title && `${title} · `}第 {source.page} 页{table ? ` · 表格第 ${table.row} 行第 ${table.column} 列` : '原文'} ↗
    </button>
    <details><summary>{table ? '查看表格证据' : source.source === 'ocr' ? '查看 OCR 引句' : '查看引句'}</summary>
      {table && <p>{table.caption}</p>}<p>{table?.model && `${table.model} · `}{table && `${table.heading} · `}{quote}</p>
    </details>
  </div>;
}

function ComparisonTable({table, evidence, openEvidence}: {table: Comparison; evidence: Evidence[]; openEvidence: (source: Evidence) => void}) {
  return <section>
    {!!table.findings?.length && <div>
      <div className="output-title"><h3>跨论文结论</h3></div>
      {table.findings.map(item => <article className="claim" key={item.label}>
        <strong>{item.label}</strong><p>{item.text}</p>
        {item.citations.map(ref => {
          const source = evidence.find(candidate => candidate.id === ref.source_id);
          return source && <SourceExcerpt key={source.id} source={source} quote={ref.quote}
            openEvidence={openEvidence} title={source.title}/>;
        })}
      </article>)}
    </div>}
    <div className="output-title"><h3>分项对比证据</h3><span>{table.papers.length} 篇</span></div>
    {table.papers.length > 2 && <p className="result-note">左右滚动可查看所有论文。</p>}
    <div className="comparison-scroll">
      <table className="comparison-table" data-wide={table.papers.length > 2}>
        <thead><tr><th scope="col">字段</th>{table.papers.map(paper => <th scope="col" key={paper.id}>{paper.title}</th>)}</tr></thead>
        <tbody>{table.rows.map(row => <tr key={row.key}>
          <th scope="row">{row.label}</th>
          {row.cells.map(cell => {
            const source = evidence.find(item => item.id === cell.source_id);
            return <td key={cell.document_id}>
              {cell.excerpts?.map(excerpt => {
                const cited = evidence.find(item => item.id === excerpt.source_id);
                return <div className="facet-excerpt" key={excerpt.label}>
                  <strong>{excerpt.label}{excerpt.display && ` · ${excerpt.display}`}</strong>
                  {cited ? <SourceExcerpt source={cited} quote={excerpt.quote} openEvidence={openEvidence}/>
                    : excerpt.alternatives ? <>
                      <p className="warning">证据冲突，待核对</p>
                      {excerpt.alternatives.map(alternative => {
                        const source = evidence.find(item => item.id === alternative.source_id);
                        return source && <div key={alternative.source_id}>
                          <strong>{alternative.label} · {alternative.display}</strong>
                          <SourceExcerpt source={source} quote={alternative.quote} openEvidence={openEvidence}/>
                        </div>;
                      })}
                    </> : <p className="result-note">未检得足够证据</p>}
                  {excerpt.searched_queries.length > 1 && <details><summary>查看补查</summary>
                    <ol>{excerpt.searched_queries.map(query => <li key={query}>{query}</li>)}</ol>
                  </details>}
                </div>;
              })}
              {cell.status === 'missing' && !cell.excerpts && <span className="result-note">未检得匹配片段</span>}
              {source && <>
                {cell.text && <p>{cell.text}</p>}
                {cell.status === 'verified_excerpt'
                  ? <><p>{cell.quote}</p>{source?.source === 'ocr' && <small className="result-note">OCR 识别片段</small>}</>
                  : <details><summary>{source.source === 'ocr' ? 'OCR 识别片段' : '原文候选'}</summary><p>{cell.quote}</p></details>}
                <button className="source-jump" onClick={() => openEvidence(source)}>第 {source.page} 页 ↗</button>
              </>}
              {(cell.searched_queries?.length ?? 0) > 1 && <details>
                <summary>查看两次检索</summary>
                <ol>{cell.searched_queries?.map((query, index) => <li key={index}>{query}</li>)}</ol>
              </details>}
            </td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
  </section>;
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [current, setCurrent] = useState<string>('');
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState('1');
  const [readingSource, setReadingSource] = useState<string>();
  const [taskPosition, setTaskPosition] = useState<{top: number; left: number}>();
  const [question, setQuestion] = useState('What hardware and training schedule were used?');
  const [targetPaper, setTargetPaper] = useState('');
  const [health, setHealth] = useState<Health | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [view, setView] = useState<View>('workspace');
  const [readerMode, setReaderMode] = useState<'pdf' | 'text'>('pdf');
  const [pageText, setPageText] = useState('');
  const [pageFromOcr, setPageFromOcr] = useState(false);
  const [duplicates, setDuplicates] = useState<Duplicate[]>([]);
  const [changingVersion, setChangingVersion] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const taskOutput = useRef<HTMLDivElement>(null);
  const paper = papers.find(x => x.id === current);
  const task = tasks.find(x => x.id === taskId);
  const activePapers = papers.filter(item => !item.archived);
  const archivedPapers = papers.filter(item => item.archived);
  const archivedCount = archivedPapers.length;
  const activeSelection = selected.filter(id => activePapers.some(p => p.id === id && ['ready','partial'].includes(p.status)));

  useLayoutEffect(() => {
    const output = taskOutput.current;
    if (view !== 'workspace' || !taskPosition || !output) return;
    output.scrollTop = taskPosition.top;
    const comparison = output.querySelector<HTMLElement>('.comparison-scroll');
    if (comparison) comparison.scrollLeft = taskPosition.left;
    setTaskPosition(undefined);
  }, [view, taskPosition]);

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
    if (current && paper && !paper.archived && !['queued', 'parsing'].includes(paper.status)) {
      api<Paper & {duplicates: Duplicate[]}>(`/documents/${current}`).then(d => {if (active) setDuplicates(d.duplicates);}).catch(() => {});
    }
    return () => {active = false;};
  }, [current, paper?.status, paper?.archived, papers.length, archivedCount]);

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

  function openReading(id: string, targetPage: number) {
    setCurrent(id); setPage(targetPage); setPageInput(String(targetPage)); setView('reading');
    setReadingSource(undefined);
  }

  function openEvidence(source: Evidence) {
    if (!papers.some(paper => paper.id === source.document_id)) {
      setError('该文献已移除，无法打开原文。');
      return;
    }
    setTaskPosition({
      top: taskOutput.current?.scrollTop ?? 0,
      left: taskOutput.current?.querySelector('.comparison-scroll')?.scrollLeft ?? 0,
    });
    openReading(source.document_id, source.page);
    setReadingSource(source.id);
  }

  function navigate(next: View) {
    if (next !== view && next !== 'workspace') setTaskPosition(undefined);
    setView(next);
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

  async function submit(compare = false) {
    setError(''); setSubmitting(true);
    try {
      const response = await api<{id: string}>(compare ? '/tasks/compare' : '/tasks', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question, document_ids: activeSelection,
          ...(compare && activeSelection.length === 1 ? {target_paper: targetPaper.trim()} : {})}),
      });
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

  async function resolveVersion(keepId: string, archiveId: string) {
    setChangingVersion(true); setError('');
    try {
      await post('/documents/resolve-version', {keep_id: keepId, archive_id: archiveId});
      setPapers(items => items.map(item => item.id === archiveId ? {...item, archived: true} : item));
      setSelected(ids => ids.filter(id => id !== archiveId));
      setDuplicates([]);
      openPaper(keepId);
      setNotice('已保留选中的版本。另一份可在文献库的「归档版本」中恢复。');
    } catch (e) {setError((e as Error).message);}
    finally {setChangingVersion(false);}
  }

  async function restoreVersion(id: string) {
    setChangingVersion(true); setError('');
    try {
      await post(`/documents/${id}/restore`, {});
      setPapers(items => items.map(item => item.id === id ? {...item, archived: false} : item));
      openPaper(id);
      setNotice('已恢复版本，可重新加入研究任务。');
    } catch (e) {setError((e as Error).message);}
    finally {setChangingVersion(false);}
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

  const eligibleCount = activeSelection.length;
  const readable = papers.filter(p => ['ready', 'partial', 'needs_ocr'].includes(p.status));
  const scopedPapers = readable.filter(p => !p.archived && (!activeSelection.length || activeSelection.includes(p.id))).slice(0, 24).map(p => p.id);
  return <div className="app">
    <aside className="nav">
      <div className="brand"><span className="brand-symbol">R<span>·</span></span><div>ResearchFlow<small>科研证据工作台</small></div></div>
      <div className="nav-label">研究空间</div>
      {(Object.keys(views) as View[]).map(key => <button key={key} className={view === key ? 'nav-item active' : 'nav-item'} onClick={() => navigate(key)}>{views[key]}</button>)}
      <div className="nav-bottom"><span className="version">LOCAL WORKSPACE / V0.1</span><p>从论文出发<br/>让每一次判断有据可查。</p><span className={health ? 'connection online' : 'connection'}>● {health ? '本地服务已连接' : '等待后端服务'}</span></div>
    </aside>
    <main>
      <header><div className="breadcrumb">我的研究 / <strong>{views[view]}</strong></div><span className="mode-badge">{health?.model_mode === 'configured' ? `模型已配置 · ${health.model_name}` : '证据检索模式 · 无模型调用'}</span></header>
      <div className="page-heading"><div><div className="eyebrow">RESEARCHFLOW</div><h1>{view === 'workspace' ? '把论文，变成可追溯的证据。' : views[view]}</h1><p>导入文献，寻找证据，回到原文。</p></div></div>
      {error && <div className="banner error" role="alert">{error}<button onClick={() => setError('')} aria-label="关闭错误提示">×</button></div>}
      {notice && <div className="banner" role="status">{notice}<button onClick={() => setNotice('')} aria-label="关闭提示">×</button></div>}
      {!health && <div className="banner error">后端尚未连接。请运行项目中的 start-backend.ps1。</div>}
      {view === 'reading' ? <Reading papers={readable} paperId={current} page={page} open={openReading} evidenceId={readingSource}
        returnToTask={taskPosition ? () => navigate('workspace') : undefined}
        returnLabel={task?.result?.comparison ? '返回对比' : '返回任务'}/>
        : view === 'discovery' ? <Discovery open={openPaper}/>
        : view === 'graph' ? <Graph ids={scopedPapers} open={openReading}/>
        : view === 'writing' ? <Writing ids={scopedPapers}/>
        : view === 'roadmap' ? <section className="roadmap">{roadmap.map(([num, title, state, desc]) => <article className="module-card" key={num}><span className="module-number">{num}</span><span className="state">{state}</span><h2>{title}</h2><p>{desc}</p></article>)}</section>
        : <div className="workspace">
        <section className="panel library"><div className="panel-heading"><div><h2>文献库 <span className="count">{activePapers.length}</span></h2><p>勾选文献，限定任务证据范围</p></div></div>
          <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple aria-label="上传PDF文件" onChange={e => void upload(e.target.files)} hidden/>
          <button className="upload" disabled={uploading || !health} onClick={() => fileInput.current?.click()}><span>＋</span>{uploading ? '上传中…' : '导入 PDF 文献'}<small>可多选 · 每份最多50MB</small></button>
          <div className="paper-list">{!papers.length ? <div className="empty small"><span>▧</span><h3>研究从一篇论文开始</h3><p>上传你的PDF，解析进度会显示在这里。</p></div> : activePapers.map(p => <article className={`paper ${current === p.id ? 'selected' : ''}`} key={p.id}>
            <div className="paper-top"><input type="checkbox" aria-label={`将${p.filename}加入任务`} checked={selected.includes(p.id)} disabled={!['ready','partial'].includes(p.status)} onChange={e => setSelected(s => e.target.checked ? [...s, p.id] : s.filter(x => x !== p.id))}/><button className="paper-title" onClick={() => openPaper(p.id)} title={p.title}>{p.title}</button></div>
            <div className="filename" title={p.filename}>{p.filename}</div><div className="paper-meta"><span className={`state ${p.status}`}>{statuses[p.status] || p.status}</span><span>{p.page_count ? `${p.processed_pages}/${p.page_count}页` : '排队中'}</span></div>
            {['queued','parsing'].includes(p.status) && <progress max={p.page_count || 1} value={p.processed_pages} aria-label="解析进度"/>}
            {p.error && <p className="inline-error">{p.error}</p>}
            <div className="paper-actions"><button onClick={() => openPaper(p.id)}>打开原文 ↗</button><button onClick={() => void remove(p.id)} aria-label={`删除${p.filename}`}>移除</button></div>
          </article>)}
          {!!archivedPapers.length && <details className="archived-papers">
            <summary>归档版本（{archivedPapers.length}）</summary>
            {archivedPapers.map(item => <article className="paper" key={item.id}>
              <button className="paper-title" onClick={() => openPaper(item.id)}>{item.title}{item.version ? ` · v${item.version}` : ''}</button>
              <p className="filename">{item.filename}</p>
              <button disabled={changingVersion} onClick={() => void restoreVersion(item.id)}>恢复到文献库</button>
            </article>)}
          </details>}</div>
          <div className="library-footer">{eligibleCount} 篇已选 · 当前为单用户本地工作区</div>
        </section>
        <section className="panel reader"><div className="panel-heading"><div><h2>原文阅读</h2><p>{paper ? paper.filename : '查看证据对应的PDF页面'}</p></div><div className="segmented"><button className={readerMode === 'pdf' ? 'chosen' : ''} onClick={() => setReaderMode('pdf')}>PDF</button><button className={readerMode === 'text' ? 'chosen' : ''} onClick={() => setReaderMode('text')}>提取文本</button></div></div>
          {paper ? <>
            <div className="reader-toolbar"><button onClick={() => jump(page-1)} disabled={page <= 1} aria-label="上一页">←</button><form onSubmit={e => {e.preventDefault(); jump(Number(pageInput));}}><input type="number" min="1" max={paper.page_count || 1} value={pageInput} onChange={e => setPageInput(e.target.value)} aria-label="PDF页码"/> / {paper.page_count || '…'} 页 <button type="submit">跳转</button></form><button onClick={() => jump(page+1)} disabled={page >= paper.page_count} aria-label="下一页">→</button><a href={`${paper.file_url}#page=${page}`} target="_blank" rel="noreferrer">新窗口 ↗</a></div>
            {paper.archived && <div className="reader-note">当前是归档版本。 <button disabled={changingVersion} onClick={() => void restoreVersion(paper.id)}>恢复到文献库</button></div>}
            {!paper.archived && duplicates.map(d => <div className="duplicate" key={d.document_id}>
              <strong>发现正文相似文献 · {Math.round(d.similarity * 100)}%</strong>
              <span>当前 {paper.filename}{paper.version ? ` · v${paper.version}` : ''}</span>
              <span>对照 {d.filename}{d.version ? ` · v${d.version}` : ''}</span>
              <span>{d.recommendation} 另一份将归档，可随时恢复。</span>
              <div className="version-actions">
                <button onClick={() => openPaper(d.document_id)}>查看对照原文 ↗</button>
                <button disabled={changingVersion} onClick={() => void resolveVersion(d.document_id, paper.id)}>保留对照版本</button>
                <button disabled={changingVersion} onClick={() => void resolveVersion(paper.id, d.document_id)}>用当前版本替换</button>
              </div>
            </div>)}
            {['needs_ocr', 'partial'].includes(paper.status) && <div className="reader-note">
              {health?.ocr_available
                ? <>仍有页面需要识别。 <button onClick={() => void retryOcr(paper.id)}>重新进行 OCR ↗</button></>
                : '仍有页面需要OCR。运行 setup-ocr.ps1 准备本地英文识别数据。'}
            </div>}
            {readerMode === 'text' && pageFromOcr && <div className="reader-note">本页文本来自 OCR，关键数字请对照 PDF 核实。</div>}
            {readerMode === 'pdf' ? <div className="reader-preview" key={`${current}:${page}`}>
              {['ready', 'partial', 'needs_ocr'].includes(paper.status)
                ? <img src={`/api/read/${paper.id}/pages/${page}/image`} alt={`${paper.filename} 第 ${page} 页`}/>
                : <p className="reader-note">{paper.status === 'failed' ? '预览不可用，可在新窗口打开 PDF。' : '解析完成后显示页面预览。'}</p>}
            </div> : <pre className="page-text">{pageText || '正在读取本页…'}</pre>}
          </> : <div className="empty reader-empty"><div className="page-illustration"><span/><span/><span/><span/><span/></div><h3>原文始终在手边</h3><p>打开一篇文献，或点击检索结果中的出处。<br/>每条证据都保留论文、页码和原文片段。</p></div>}
        </section>
        <section className="panel agent"><div className="panel-heading"><div><h2>研究任务</h2><p>限定证据范围，让结果可核查</p></div><span className="tiny-mark">RF</span></div>
          <div className="task-compose"><label htmlFor="question">你希望从论文中找到什么？</label><textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} maxLength={2000}/><div className="examples"><button onClick={() => setQuestion('What hardware and training schedule were used?')}>训练硬件</button><button onClick={() => setQuestion('这篇论文的 Adam β₁ 和 β₂ 分别是多少？')}>Adam β 参数</button><button onClick={() => setQuestion('Compare the BLEU results for English to German and English to French.')}>实验指标</button></div><button className="primary" onClick={() => void submit()} disabled={!health || !eligibleCount || question.trim().length < 2 || submitting}>{submitting ? '提交中…' : health?.model_mode === 'configured' ? '执行研究任务 →' : '检索原文证据 →'}</button><p className="compose-note">{health?.model_mode === 'configured' ? '使用后端配置的模型；引用校验通过仍需人工核对。' : '大模型已关闭。可检索原文，并从单篇论文直接核对 Adam β 参数。'}</p></div>
          <details className="task-compose compare-form">
            <summary>补齐论文并对比</summary>
            <p className="compose-note">可勾选 2–4 篇直接对比。只勾选一篇时，系统会按 BERT、arXiv 编号或《完整标题》补齐目标论文。</p>
            {eligibleCount <= 1 && <>
              <label htmlFor="target-paper">目标论文（可选，优先使用此栏）</label>
              <input id="target-paper" value={targetPaper} onChange={e => setTargetPaper(e.target.value)} maxLength={240}/>
            </>}
            <div className="examples"><button onClick={() => {
              setTargetPaper('');
              setQuestion('BERT 继承了 Transformer Encoder 的什么，又改了什么？');
            }} disabled={eligibleCount > 1}>填入 BERT 架构示例</button>
              {eligibleCount >= 1 && eligibleCount <= 2 && <button onClick={() => {
                setTargetPaper('');
                setQuestion('Transformer Big 和 BERT Large 各用了什么设备，训练了多久？能据此判断哪个模型训练效率更高吗？');
              }}>填入训练实验示例</button>}
              {eligibleCount === 3 && <button onClick={() => setQuestion('比较 GPT-1、GPT-2、GPT-3 的参数量、上下文窗口和 LN 位置。')}>填入 GPT 三代示例</button>}
            </div>
            <button className="primary" onClick={() => void submit(true)}
              disabled={!health || eligibleCount < 1 || eligibleCount > 4 || question.trim().length < 2 || submitting}>
              {eligibleCount > 1 ? '比较所选论文 →' : '补齐并对比 →'}
            </button>
          </details>
          {tasks.length > 0 && <div className="task-history"><label htmlFor="history">任务记录</label><select id="history" value={taskId} onChange={e => setTaskId(e.target.value)}><option value="">选择已保存的任务</option>{tasks.map(t => <option key={t.id} value={t.id}>{t.question.slice(0,55)} · {t.status}</option>)}</select></div>}
          <div className="task-output" ref={taskOutput} aria-live="polite">{task ? <>
            <div className="output-title"><h3>执行记录</h3><span className={`state ${task.status}`}>{({queued:'排队中',running:'执行中',completed:'已完成',failed:'失败'} as Record<string,string>)[task.status]}</span></div>
            <ol className="timeline">{task.events.map((event,i) => <li key={i}><strong>{event.tool}</strong><span>{event.message}</span></li>)}</ol>
            {task.result?.warnings.map((w,i) => <p className="warning" key={i}>{w}</p>)}
            {!!task.result?.claims.length && <><h3>有来源的结论</h3>{task.result.claims.map((claim,i) => {const source = task.result?.evidence.find(x => x.id === claim.source_id); return <article className="claim" key={i}><p>{claim.text}</p><blockquote>{claim.quote}</blockquote>{source && <button onClick={() => openEvidence(source)}>原文 · 第{source.page}页 ↗</button>}</article>;})}</>}
            {task.result?.comparison ? <>
              <ComparisonTable table={task.result.comparison} evidence={task.result.evidence} openEvidence={openEvidence}/>
              <p><a href={`/api/tasks/${task.id}/report.md`}>下载研究报告 · Markdown</a></p>
            </> : task.result && task.result.mode !== 'error' && task.document_ids.length > 1 ? <>
              <div className="output-title"><h3>多文献证据对照</h3><span>{task.document_ids.length} 篇</span></div>
              <p className="result-note">各篇按关键词相关性排序。点击页码核对片段。</p>
              {task.document_ids.map(id => {
                const sources = task.result?.evidence.filter(source => source.document_id === id) ?? [];
                const filename = papers.find(p => p.id === id)?.filename ?? sources[0]?.filename ?? '已移除文献';
                return <section key={id}>
                  <div className="output-title"><h3>{filename}</h3><span>{sources.length} 条</span></div>
                  {sources.length ? <EvidenceCards sources={sources} openEvidence={openEvidence}/> : <p className="result-note">这篇文献未返回匹配片段。</p>}
                </section>;
              })}
            </> : !!task.result?.evidence.length && <>
              <div className="output-title"><h3>原文候选</h3><span>{task.result.evidence.length} 条</span></div>
              <p className="result-note">关键词相关性排序。点击页码核对片段。</p>
              <EvidenceCards sources={task.result.evidence} openEvidence={openEvidence}/>
            </>}
          </> : <div className="empty small task-empty"><span>◎</span><h3>让证据先行</h3><p>在左侧勾选已解析文献，再发起任务。<br/>执行步骤与来源将保存在这里。</p></div>}</div>
        </section>
      </div>}
      <footer>ResearchFlow / 从原文到证据，再到判断。<span>开发用公开替代数据集 · 不代表官方验收结果</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);

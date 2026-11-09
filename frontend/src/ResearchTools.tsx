import { useEffect, useRef, useState } from 'react';
import type { UIEvent } from 'react';
import { api, post } from './api';

type OpenPaper = (id: string, page: number) => void;
type Paper = {id: string; title: string; filename: string; page_count: number; status: string};
type Ref = {label: string; page: number; text: string; citation_page: number | null};
type Index = {title: string; metadata: {author: string}; abstract: string; outline: {title: string; page: number; level: number}[]; references: Ref[]};
type Page = {text: string; source: string; translation: string | null; citations: string[]};
type Structure = {tables: {rows: (string | null)[][]}[]; figures: {bbox: number[]}[]; formula_candidates: {text: string}[]};
type TaskStatus = {status: string; events: {message: string}[]; result?: {warnings: string[]}};

function message(error: unknown) {
  return error instanceof Error ? error.message : '请求未完成。';
}

function saveText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], {type:'text/plain;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ErrorNote({error}: {error: string}) {
  return error ? <p className="banner error" role="alert">{error}</p> : null;
}

export function Reading({papers, paperId, page, open}: {papers: Paper[]; paperId: string; page: number; open: OpenPaper}) {
  const paper = papers.find(item => item.id === paperId) ?? papers[0];
  const [index, setIndex] = useState<Index>();
  const [data, setData] = useState<Page>();
  const [structure, setStructure] = useState<Structure>();
  const [reference, setReference] = useState<Ref>();
  const [returnPage, setReturnPage] = useState<number>();
  const [translation, setTranslation] = useState('');
  const [job, setJob] = useState('');
  const [progress, setProgress] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const original = useRef<HTMLDivElement>(null);
  const translated = useRef<HTMLDivElement>(null);
  const scrolling = useRef(false);
  const pageNumber = Math.max(1, Math.min(page, paper?.page_count || 1));

  useEffect(() => {
    let live = true;
    setIndex(undefined);
    setReference(undefined);
    setReturnPage(undefined);
    if (paper?.id) api<Index>(`/read/${paper.id}`).then(value => {if (live) setIndex(value);}).catch(e => {if (live) setError(message(e));});
    return () => { live = false; };
  }, [paper?.id]);

  useEffect(() => {
    let live = true;
    setData(undefined);
    setStructure(undefined);
    setTranslation('');
    if (paper?.id) api<Page>(`/read/${paper.id}/pages/${pageNumber}`).then(value => {if (live) setData(value);}).catch(e => {if (live) setError(message(e));});
    return () => { live = false; };
  }, [paper?.id, pageNumber, refresh]);

  useEffect(() => {
    if (!job) return;
    let live = true;
    const poll = async () => {
      try {
        const task = await api<TaskStatus>(`/tasks/${job}`);
        if (!live) return;
        setProgress(task.events.at(-1)?.message ?? '翻译排队中…');
        if (task.status === 'completed' || task.status === 'failed') {
          setJob('');
          setRefresh(count => count + 1);
          if (task.status === 'failed') setError(task.result?.warnings[0] ?? '翻译失败。');
        }
      } catch (e) {if (live) {setError(message(e)); setJob('');}}
    };
    const timer = window.setInterval(() => void poll(), 1500);
    void poll();
    return () => { live = false; window.clearInterval(timer); };
  }, [job]);

  if (!paper) return <p className="tool-box">上传并解析一篇文献后，可以在这里联动阅读。</p>;

  async function queueTranslation(id: string, start: number, end: number) {
    setError(''); setBusy(true);
    try {
      setJob((await post<{id: string}>(`/read/${id}/translate`, {start,end})).id);
      setProgress('翻译排队中…');
    } catch(e) {setError(message(e));}
    finally {setBusy(false);}
  }

  async function translateSelection() {
    const selection = window.getSelection();
    if (!selection || !original.current?.contains(selection.anchorNode) || !original.current.contains(selection.focusNode) || !selection.toString().trim()) {
      setError('请先在左侧原文文本中划选需要翻译的内容。');
      return;
    }
    setError(''); setBusy(true);
    try {setTranslation((await post<{translation: string}>('/translate', {text: selection.toString().trim(), language:'Chinese'})).translation);}
    catch(e) {setError(message(e));}
    finally {setBusy(false);}
  }

  function sync(event: UIEvent<HTMLDivElement>, other: HTMLDivElement | null) {
    if (scrolling.current || !other) return;
    const source = event.currentTarget;
    const available = source.scrollHeight - source.clientHeight;
    if (available <= 0) return;
    const target = source.scrollTop / available * (other.scrollHeight - other.clientHeight);
    if (Math.abs(other.scrollTop - target) < 2) return;
    scrolling.current = true;
    other.scrollTop = target;
    requestAnimationFrame(() => {scrolling.current = false;});
  }

  function jumpReference(ref: Ref) {
    setReference(ref);
    setReturnPage(pageNumber);
    open(paper!.id, ref.page);
  }

  return <section className="tool-box">
    <h2>联动阅读与翻译</h2>
    <div className="tool-actions">
      <select aria-label="阅读文献" value={paper.id} onChange={e => open(e.target.value,1)}>{papers.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
      <button disabled={pageNumber <= 1} onClick={() => open(paper.id,pageNumber-1)}>上一页</button>
      <span>{pageNumber} / {paper.page_count}</span>
      <button disabled={pageNumber >= paper.page_count} onClick={() => open(paper.id,pageNumber+1)}>下一页</button>
      <select aria-label="目录" value="" onChange={e => open(paper.id,Number(e.target.value))}>
        <option value="" disabled>目录跳转</option>
        {index?.outline.map((item,i) => <option key={i} value={item.page}>{'　'.repeat(Math.min(2,item.level-1))}{item.title} · {item.page}</option>)}
      </select>
    </div>
    <ErrorNote error={error}/>
    <div className="tool-actions">
      <button disabled={busy || !!job} onMouseDown={e => e.preventDefault()} onClick={() => void translateSelection()}>翻译划选</button>
      <button disabled={busy || !!job} onClick={() => void queueTranslation(paper.id,pageNumber,pageNumber)}>翻译本页</button>
      <button disabled={busy || !!job} onClick={() => void queueTranslation(paper.id,1,paper.page_count)}>翻译整篇</button>
      <a href={`/api/read/${paper.id}/translation.md`}>导出已完成译文</a>
      <button onClick={() => void api<Structure>(`/read/${paper.id}/pages/${pageNumber}/structure`).then(setStructure).catch(e => setError(message(e)))}>查看页面结构</button>
    </div>
    {progress && <p className="compose-note" role="status">{progress}</p>}
    {index?.abstract && <details><summary>摘要与元信息</summary><p>{index.metadata.author}</p><p>{index.abstract}</p></details>}
    {structure && <details open><summary>当前页结构</summary>
      <p>{structure.tables.length} 个表格，{structure.figures.length} 个图像区域，{structure.formula_candidates.length} 个公式候选。</p>
      {structure.tables.map((table,i) => <Grid key={i} rows={table.rows}/>)}
      {structure.formula_candidates.map((item,i) => <pre key={i}>{item.text}</pre>)}
    </details>}
    <div className="tool-columns">
      <div><h3>PDF 原版与文本</h3><div className="read-scroll" ref={original} onScroll={event => sync(event, translated.current)}>
        <img src={`/api/read/${paper.id}/pages/${pageNumber}/image`} alt={`原始 PDF 第 ${pageNumber} 页`}/>
        <pre>{data?.text || '本页暂无可提取文本。'}</pre>
      </div></div>
      <div><h3>中文译文</h3><div className="read-scroll" ref={translated} onScroll={event => sync(event, original.current)}>
        <pre>{translation || data?.translation || '翻译任务完成后，译文会显示在这里。'}</pre>
      </div></div>
    </div>
    <div className="tool-actions">{data?.citations.map(label => {
      const item = index?.references.find(ref => ref.label === label);
      return item && <button key={label} onClick={() => jumpReference(item)}>引用 {label}</button>;
    })}</div>
    {reference && <div className="tool-result"><p>{reference.text}</p><button onClick={() => open(paper.id,returnPage || reference.citation_page || 1)}>返回正文</button></div>}
    <details><summary>参考文献（{index?.references.length ?? 0} 项或页）</summary>
      {index?.references.map((ref,i) => <div className="tool-result" key={i}><p>{ref.text.slice(0,700)}</p>
        <button onClick={() => jumpReference(ref)}>参考文献第 {ref.page} 页</button>
        {ref.citation_page && <button onClick={() => open(paper.id,ref.citation_page!)}>正文第 {ref.citation_page} 页</button>}
      </div>)}
    </details>
  </section>;
}

type SearchPaper = {id: string; title: string; authors: string[]; published: string; abstract: string; url: string; code_url?: string; rating?: number; reason?: string};
type Topic = {id: string; query: string; interval_minutes: number; error?: string; results: SearchPaper[]};

export function Discovery({open}: {open: OpenPaper}) {
  const [query,setQuery] = useState('');
  const [results,setResults] = useState<SearchPaper[]>([]);
  const [topics,setTopics] = useState<Topic[]>([]);
  const [interval,setIntervalMinutes] = useState(60);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const update = (topic: Topic) => setTopics(items => [...items.filter(item => item.id !== topic.id),topic]);

  useEffect(() => {
    let live = true;
    const load = () => api<Topic[]>('/literature/topics').then(items => {if (live) setTopics(items);}).catch(e => {if (live) setError(message(e));});
    void load();
    const timer = window.setInterval(load, 20000);
    return () => {live = false; window.clearInterval(timer);};
  }, []);

  async function action(operation: () => Promise<void>) {
    setBusy(true); setError('');
    try {await operation();} catch(e) {setError(message(e));} finally {setBusy(false);}
  }

  function card(paper: SearchPaper, topic?: Topic) {
    return <article key={paper.id} className="tool-result">
      <h3><a href={paper.url} target="_blank" rel="noreferrer">{paper.title}</a></h3>
      <p>{paper.authors.slice(0,4).join('、')} · {paper.published}</p>
      {paper.reason && <p>{paper.reason}</p>}
      <details><summary>摘要</summary><p>{paper.abstract}</p></details>
      <div className="tool-actions">
        <button disabled={busy} onClick={() => void action(async () => {
          const imported = await post<{id:string}>('/literature/import',{arxiv_id:paper.id}); open(imported.id,1);
        })}>导入 PDF</button>
        {paper.code_url && <a href={paper.code_url} target="_blank" rel="noreferrer">摘要中的代码链接</a>}
        {topic && [1,-1].map(rating => <button key={rating} disabled={busy} aria-pressed={paper.rating===rating} onClick={() => void action(async () => {
          update(await post<Topic>(`/literature/topics/${topic.id}/feedback`,{paper_id:paper.id,rating:paper.rating===rating ? 0 : rating}));
        })}>{rating===1 ? '赞' : '踩'}</button>)}
      </div>
    </article>;
  }

  return <section className="tool-box">
    <h2>arXiv 检索与主题追踪</h2>
    <ErrorNote error={error}/>
    <div className="tool-actions">
      <input aria-label="arXiv 检索词" value={query} maxLength={240} placeholder="标题、英文关键词或 arXiv 编号" onChange={e => setQuery(e.target.value)}/>
      <button disabled={busy || !query.trim()} onClick={() => void action(async () => setResults(await api<SearchPaper[]>(`/literature/search?query=${encodeURIComponent(query)}`)))}>检索</button>
      <label>刷新间隔 <select value={interval} onChange={e => setIntervalMinutes(Number(e.target.value))}>
        <option value={30}>30 分钟</option><option value={60}>1 小时</option><option value={1440}>每天</option>
      </select></label>
      <button disabled={busy || !query.trim()} onClick={() => void action(async () => {
        const topic = await post<Topic>('/literature/topics',{query,interval_minutes:interval});
        update(topic);
        update(await post<Topic>(`/literature/topics/${topic.id}/refresh`,{}));
      })}>追踪并刷新</button>
    </div>
    {results.map(item => card(item))}
    {topics.map(topic => <section key={topic.id} className="tool-result">
      <h3>{topic.query} · 每 {topic.interval_minutes} 分钟</h3>
      <div className="tool-actions">
        <button disabled={busy} onClick={() => void action(async () => update(await post<Topic>(`/literature/topics/${topic.id}/refresh`,{})))}>立即刷新</button>
        <button disabled={busy} onClick={() => void action(async () => {
          await api(`/literature/topics/${topic.id}`,{method:'DELETE'}); setTopics(items => items.filter(item => item.id!==topic.id));
        })}>停止追踪</button>
      </div>
      {topic.error && <ErrorNote error={topic.error}/>}
      {topic.results.map(item => card(item,topic))}
    </section>)}
  </section>;
}

type GraphData = {nodes: (Pick<Paper,'id'|'title'> & {rank: number})[]; edges: {source: string; target: string; page: number; quote: string}[];
  overview: string[]; future_work: {document_id: string; page: number; quote: string; related: {document_id: string; title: string; page: number}[]}[]};

export function Graph({ids,open}: {ids: string[]; open: OpenPaper}) {
  const [data,setData] = useState<GraphData>();
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const points = new Map(data?.nodes.map((node,i) => {
    const angle = 2*Math.PI*i/data.nodes.length;
    return [node.id,{x:320+215*Math.cos(angle),y:185+134*Math.sin(angle)}];
  }));
  const title = (id: string) => data?.nodes.find(node => node.id === id)?.title ?? id;
  return <section className="tool-box">
    <h2>局域引用图谱</h2>
    <div className="tool-actions"><span>当前纳入 {ids.length} 篇。可在文献库中勾选范围。</span>
      <button disabled={busy || !ids.length} onClick={async () => {
        setBusy(true); setError('');
        try {setData(await post('/graph',{document_ids:ids}));} catch(e) {setError(message(e));} finally {setBusy(false);}
      }}>生成图谱</button>
    </div>
    <ErrorNote error={error}/>
    {data && <>
      <div className="tool-columns">
        <svg className="tool-figure" viewBox="0 0 640 370" aria-label="局域引用图谱">
          <defs><marker id="citation-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#83a06a"/></marker></defs>
          {data.edges.map((edge,i) => {
            const a = points.get(edge.source)!, b = points.get(edge.target)!;
            const length = Math.hypot(b.x-a.x,b.y-a.y) || 1;
            return <line key={i} x1={a.x} y1={a.y} x2={b.x-18*(b.x-a.x)/length} y2={b.y-18*(b.y-a.y)/length} stroke="#83a06a" strokeWidth="2" markerEnd="url(#citation-arrow)"/>;
          })}
          {data.nodes.map((node,i) => {
            const point = points.get(node.id)!;
            return <g key={node.id}><circle cx={point.x} cy={point.y} r="16" fill="#31573e"/><text x={point.x} y={point.y+4} textAnchor="middle" fill="#fff" fontSize="10">{i+1}</text></g>;
          })}
        </svg>
        <div><h3>库内 PageRank</h3>{data.nodes.map((node,i) => <p key={node.id}><button onClick={() => open(node.id,1)}>{i+1}. {node.title} · {node.rank.toFixed(3)}</button></p>)}</div>
      </div>
      <h3>引用证据与简述</h3>
      {!data.edges.length && <p>当前范围内没有找到可核对的引用边。</p>}
      {data.edges.map((edge,i) => <details className="tool-result" key={i}><summary>{title(edge.source)} → {title(edge.target)}</summary>
        <p>{edge.quote.slice(0,750)}</p><button onClick={() => open(edge.source,edge.page)}>核对第 {edge.page} 页</button>
      </details>)}
      {data.overview.map((line,i) => <p key={i}>{line}</p>)}
      <h3>未来工作线索</h3>
      {data.future_work.map((item,i) => <article className="tool-result" key={i}><p>{item.quote}</p>
        <button onClick={() => open(item.document_id,item.page)}>原文第 {item.page} 页</button>
        {item.related.map(related => <button key={related.document_id} onClick={() => open(related.document_id,related.page)}>后续引用：{related.title}</button>)}
      </article>)}
      {!data.future_work.length && <p>当前范围内未提取到明确的未来工作段落。</p>}
    </>}
  </section>;
}

type Chart = {columns: string[]; numeric_columns: string[]; label: string; value: string; labels: string[]; values: number[]; kind: 'bar'|'line'|'radar'; rows: string[][]; caption: string; script: string};

function Grid({headers,rows}: {headers?: string[]; rows: (string | null)[][]}) {
  return <div className="comparison-scroll"><table className="comparison-table">
    {headers && <thead><tr>{headers.map((header,i) => <th key={i}>{header}</th>)}</tr></thead>}
    <tbody>{rows.slice(0,20).map((row,i) => <tr key={i}>{row.map((cell,j) => <td key={j}>{cell}</td>)}</tr>)}</tbody>
  </table></div>;
}

function ChartPreview({chart}: {chart: Chart}) {
  const values = chart.values.slice(0,60);
  const low = Math.min(0,...values), high = Math.max(0,...values), span = high-low || 1;
  const y = (value: number) => 220-(value-low)*170/span;
  const step = 570/values.length;
  const points = values.map((value,i) => {
    if (chart.kind !== 'radar') return [54+step*(i+.5), y(value)];
    const angle = i*2*Math.PI/values.length;
    return [330+160*Math.cos(angle),170+58*Math.sin(angle)-85*(value-low)/span];
  });
  return <svg className="tool-figure" viewBox="0 0 660 255" aria-label={chart.caption}>
    {chart.kind !== 'radar' && <line x1="54" y1={y(0)} x2="624" y2={y(0)} stroke="#83947d"/>}
    {chart.kind === 'bar' ? points.map(([x,top],i) => <rect key={i} x={x-step*.32} y={Math.min(top,y(0))} width={step*.64} height={Math.max(1,Math.abs(top-y(0)))} fill="#62824b"><title>{chart.labels[i]}：{values[i]}</title></rect>)
      : <><polyline fill={chart.kind==='radar' ? '#d4e4c8' : 'none'} stroke="#47724b" strokeWidth="2" points={(chart.kind==='radar' ? [...points,points[0]] : points).map(point => point.join(',')).join(' ')}/>
        {points.map(([x,top],i) => <circle key={i} cx={x} cy={top} r="3" fill="#31573e"><title>{chart.labels[i]}：{values[i]}</title></circle>)}</>}
    {points.length <= 14 && points.map(([x,top],i) => <text key={i} x={x} y={chart.kind==='radar' ? top-10 : 243} textAnchor="middle" fontSize="10">{chart.labels[i].slice(0,12)}</text>)}
  </svg>;
}

export function Writing({ids}: {ids: string[]}) {
  const [idea,setIdea] = useState('');
  const [outline,setOutline] = useState<{markdown: string; mermaid: string}>();
  const [translation,setTranslation] = useState('');
  const [file,setFile] = useState<File>();
  const [chart,setChart] = useState<Chart>();
  const [kind,setKind] = useState<Chart['kind']>('bar');
  const [label,setLabel] = useState('');
  const [value,setValue] = useState('');
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  async function action(operation: () => Promise<void>) {
    setBusy(true); setError('');
    try {await operation();} catch(e) {setError(message(e));} finally {setBusy(false);}
  }
  return <section className="tool-box">
    <h2>写作与绘图</h2>
    <ErrorNote error={error}/>
    <label>研究构想或待译中文摘要<textarea value={idea} onChange={e => setIdea(e.target.value)} maxLength={2500}/></label>
    <div className="tool-actions">
      <button disabled={busy || idea.trim().length < 2 || !ids.length} onClick={() => void action(async () => setOutline(await post('/write/outline',{idea,document_ids:ids})))}>用 {ids.length} 篇文献生成提纲</button>
      <button disabled={busy || !idea.trim()} onClick={() => void action(async () => setTranslation((await post<{translation:string}>('/translate',{text:idea,language:'English'})).translation))}>学术英文翻译</button>
    </div>
    {outline && <div className="tool-result"><pre>{outline.markdown}</pre><div className="tool-actions">
      <button onClick={() => saveText('outline.md',outline.markdown)}>下载提纲</button>
      <button onClick={() => saveText('architecture.mmd',outline.mermaid)}>下载 Mermaid 架构图</button>
    </div></div>}
    {translation && <pre className="tool-result">{translation}</pre>}
    <h3>CSV 图表与图注</h3>
    <div className="tool-actions">
      <input aria-label="上传 CSV" type="file" accept=".csv,text/csv" onChange={e => {setFile(e.target.files?.[0]); setChart(undefined); setLabel(''); setValue('');}}/>
      <select aria-label="图表类型" value={kind} onChange={e => setKind(e.target.value as Chart['kind'])}><option value="bar">柱状图</option><option value="line">折线图</option><option value="radar">三维雷达图</option></select>
      {chart && <><select aria-label="标签列" value={label || chart.label} onChange={e => setLabel(e.target.value)}>{chart.columns.map(name => <option key={name}>{name}</option>)}</select>
        <select aria-label="数值列" value={value || chart.value} onChange={e => setValue(e.target.value)}>{chart.numeric_columns.map(name => <option key={name}>{name}</option>)}</select></>}
      <button disabled={busy || !file} onClick={() => void action(async () => {
        const form = new FormData(); form.append('file',file!); form.append('kind',kind); form.append('label',label); form.append('value',value);
        setChart(await api<Chart>('/write/chart',{method:'POST',body:form}));
      })}>生成图表</button>
    </div>
    {chart && <><ChartPreview chart={chart}/><p>{chart.caption}</p>
      <button onClick={() => saveText('draw_chart.py',chart.script)}>下载 Matplotlib 脚本</button>
      <details><summary>查看数据表（前 20 行）</summary><Grid headers={chart.columns} rows={chart.rows}/></details>
    </>}
  </section>;
}

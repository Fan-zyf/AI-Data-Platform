/**
 * 简单的类聊天气泡：每条消息 = 一次 Q&A。
 * - 用户问题用 right-aligned 蓝紫色气泡
 * - Agent 回答用 left-aligned 深色卡片
 */
export default function AgentChat({ messages }) {
  if (!messages || !messages.length) {
    return <p className="panel-empty">还没有对话记录，输入问题后点击「开始分析」即可体验 AI Data Analyst Agent。</p>
  }
  return (
    <div className="agent-chat">
      {messages.map((msg, i) => (
        <Message key={i} message={msg} />
      ))}
    </div>
  )
}

function Message({ message }) {
  if (message.role === 'user') {
    return (
      <div className="agent-bubble agent-bubble-user">
        <div className="agent-bubble-meta">您</div>
        <div className="agent-bubble-body">{message.text}</div>
      </div>
    )
  }
  if (message.role === 'assistant') {
    const r = message.payload || {}
    return (
      <div className="agent-bubble agent-bubble-assistant">
        <div className="agent-bubble-meta">
          AI Data Analyst
          {r.llm?.is_mock ? <span className="ml-tip"> · Mock 模式</span> : null}
        </div>
        <div className="agent-bubble-body">
          {r.answer ? (
            <MarkdownText text={r.answer} />
          ) : (
            <p className="panel-empty">未生成回答。</p>
          )}
        </div>
      </div>
    )
  }
  return null
}

function MarkdownText({ text }) {
  // 极简 Markdown 渲染：仅支持 ### 标题 与 - 列表
  const lines = String(text || '').split('\n')
  const blocks = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.startsWith('### ')) {
      blocks.push(<h6 key={`h-${i}`}>{line.slice(4).trim()}</h6>)
      i += 1
      continue
    }
    if (line.startsWith('- ')) {
      const items = []
      while (i < lines.length && lines[i].startsWith('- ')) {
        items.push(lines[i].slice(2))
        i += 1
      }
      blocks.push(
        <ul key={`u-${i}`}>
          {items.map((it, k) => (
            <li key={k}>{it}</li>
          ))}
        </ul>,
      )
      continue
    }
    if (line.trim() === '') {
      i += 1
      continue
    }
    blocks.push(<p key={`p-${i}`}>{line}</p>)
    i += 1
  }
  return <div className="agent-markdown">{blocks}</div>
}

import { useState } from 'react'
import { useWS } from './hooks/useWS'
import Header       from './components/Header'
import TopologyPanel from './components/TopologyPanel'
import TabBar        from './components/TabBar'
import TrainingTab   from './components/TrainingTab'
import TimelineTab   from './components/TimelineTab'
import BatchesTab    from './components/BatchesTab'
import { C } from './theme'

export default function App() {
  const { data, connected, lastUpdate } = useWS()
  const [tab, setTab]          = useState('Training')
  const [jumpBatchId, setJump] = useState(null)

  function handleSelectBatch(bid) {
    setJump(bid)
    setTab('Timeline')
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: C.bg, color: C.text }}>
      <Header connected={connected} lastUpdate={lastUpdate} epochs={data.epochs} batches={data.batches} />
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <TopologyPanel topology={data.topology} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
          <TabBar active={tab} onChange={setTab} />
          {tab === 'Training' && <TrainingTab  epochs={data.epochs} batches={data.batches} topology={data.topology} />}
          {tab === 'Timeline' && <TimelineTab  batches={data.batches} topology={data.topology} jumpBatchId={jumpBatchId} onJumpConsumed={() => setJump(null)} />}
          {tab === 'Batches'  && <BatchesTab   batches={data.batches} topology={data.topology} onSelectBatch={handleSelectBatch} />}
        </div>
      </div>
    </div>
  )
}

import { useState, useEffect } from 'react'

const EMPTY = { topology: [], batches: [], epochs: [] }

export function useWS() {
  const [data, setData]             = useState(EMPTY)
  const [connected, setConnected]   = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)

  useEffect(() => {
    let ws, timer

    function connect() {
      ws = new WebSocket(`ws://${location.host}/ws`)
      ws.onopen    = () => setConnected(true)
      ws.onmessage = e => { setData(JSON.parse(e.data)); setLastUpdate(new Date().toLocaleTimeString()) }
      ws.onclose   = () => { setConnected(false); timer = setTimeout(connect, 3000) }
      ws.onerror   = () => ws.close()
    }

    connect()
    return () => { clearTimeout(timer); ws?.close() }
  }, [])

  return { data, connected, lastUpdate }
}

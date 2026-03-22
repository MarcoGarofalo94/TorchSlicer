export const C = {
  bg:       '#0d1117',
  surface:  '#161b22',
  surface2: '#1c2128',
  border:   '#30363d',
  text:     '#e6edf3',
  muted:    '#8b949e',
  blue:     '#58a6ff',
  orange:   '#f0883e',
  green:    '#3fb950',
  purple:   '#bc8cff',
  red:      '#f85149',
  yellow:   '#e3b341',
  mono:     "'SF Mono','Fira Code','Cascadia Code',monospace",
}

export const panel = {
  background:    C.surface,
  border:        `1px solid ${C.border}`,
  borderRadius:  8,
  padding:       14,
  overflow:      'hidden',
  display:       'flex',
  flexDirection: 'column',
}

export const panelTitle = {
  fontSize:      10,
  fontWeight:    700,
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color:         C.muted,
  marginBottom:  10,
  flexShrink:    0,
}

export const selectStyle = {
  background:   C.surface,
  color:        C.text,
  border:       `1px solid ${C.border}`,
  borderRadius: 5,
  padding:      '4px 10px',
  fontSize:     11,
  cursor:       'pointer',
}

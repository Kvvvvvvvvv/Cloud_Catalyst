import axios from 'axios'


const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'


export const runSimulation = (payload) => axios.post(`${API_BASE}/run-simulation`, payload, { timeout: 120000 })
export const predict = (symbol, days=30) => axios.get(`${API_BASE}/predict/${encodeURIComponent(symbol)}?days=${days}`)
export const optimize = (payload) => axios.post(`${API_BASE}/optimize`, payload)
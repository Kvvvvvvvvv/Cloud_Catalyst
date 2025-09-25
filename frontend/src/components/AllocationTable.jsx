import React from "react";
import Plot from "react-plotly.js";

const colorPalette = [
  "#1a73e8",
  "#34a853",
  "#fbbc04",
  "#ea4335",
  "#9c27b0",
  "#00acc1",
  "#ff7043",
  "#5e35b1",
  "#43a047",
  "#d81b60",
];

export default function AllocationTable({ allocation }) {
  const rows = (allocation || []).filter((item) =>
    Number.isFinite(Number(item.allocation_pct))
  );

  if (!rows.length) {
    return null;
  }

  const values = rows.map((row) => Number(row.allocation_pct));
  const labels = rows.map((row) => row.asset);

  return (
    <div className="card allocation-card">
      <h3>Allocation Breakdown</h3>
      <div className="allocation-layout">
        <div className="allocation-chart">
          <Plot
            data={[
              {
                values,
                labels,
                type: "pie",
                hole: 0.45,
                textinfo: "label+percent",
                hoverinfo: "label+percent",
                marker: { colors: colorPalette },
              },
            ]}
            layout={{
              height: 320,
              margin: { t: 10, b: 10, l: 10, r: 10 },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
        <div className="allocation-table-wrapper">
          <table className="data-table allocation-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Allocation</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.asset}>
                  <td>{row.asset}</td>
                  <td>{Number(row.allocation_pct).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

export default function Charts({ portfolioValues, predictions }) {
  const portfolioReady = Boolean(
    portfolioValues?.dates?.length && portfolioValues?.values?.length
  );

  const predictionItems = useMemo(
    () =>
      (predictions || []).filter(
        (item) => item?.ds?.length && item?.yhat?.length && !item.error
      ),
    [predictions]
  );

  const [activeSymbol, setActiveSymbol] = useState(
    predictionItems[0]?.symbol || null
  );

  useEffect(() => {
    if (!predictionItems.find((item) => item.symbol === activeSymbol)) {
      setActiveSymbol(predictionItems[0]?.symbol || null);
    }
  }, [predictionItems, activeSymbol]);

  const activePrediction = predictionItems.find(
    (item) => item.symbol === activeSymbol
  );

  const recentRows = useMemo(() => {
    if (!activePrediction) {
      return [];
    }
    const { ds, yhat, yhat_lower, yhat_upper } = activePrediction;
    const length = ds.length;
    const start = Math.max(0, length - 10);
    return ds.slice(start).map((date, idx) => ({
      date,
      yhat: yhat[start + idx],
      lower: yhat_lower[start + idx],
      upper: yhat_upper[start + idx],
    }));
  }, [activePrediction]);

  return (
    <div className="charts-stack">
      {portfolioReady && (
        <div className="card">
          <h3>Portfolio Value Projection</h3>
          <Plot
            data={[
              {
                x: portfolioValues.dates,
                y: portfolioValues.values,
                type: "scatter",
                mode: "lines",
                line: { color: "#1a73e8", width: 3 },
                hovertemplate: "%{x}<br>$%{y:,.2f}<extra></extra>",
              },
            ]}
            layout={{
              height: 360,
              margin: { t: 40, r: 20, b: 40, l: 60 },
              yaxis: { title: "Value ($)", tickprefix: "$" },
              xaxis: { title: "Date" },
              hovermode: "x unified",
              template: "plotly_white",
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      )}

      {predictionItems.length > 0 && (
        <div className="card prediction-card">
          <div className="tab-bar">
            {predictionItems.map((prediction) => (
              <button
                key={prediction.symbol}
                type="button"
                className={`tab-button ${
                  prediction.symbol === activeSymbol ? "active" : ""
                }`}
                onClick={() => setActiveSymbol(prediction.symbol)}
              >
                {prediction.symbol}
              </button>
            ))}
          </div>

          {activePrediction ? (
            <>
              <Plot
                data={[
                  {
                    x: activePrediction.ds,
                    y: activePrediction.yhat,
                    type: "scatter",
                    mode: "lines",
                    name: "Predicted Price",
                    line: { color: "#34a853", width: 3 },
                  },
                  {
                    x: activePrediction.ds,
                    y: activePrediction.yhat_upper,
                    type: "scatter",
                    mode: "lines",
                    line: { width: 0 },
                    showlegend: false,
                    hoverinfo: "skip",
                  },
                  {
                    x: activePrediction.ds,
                    y: activePrediction.yhat_lower,
                    type: "scatter",
                    mode: "lines",
                    line: { width: 0 },
                    fill: "tonexty",
                    fillcolor: "rgba(52, 168, 83, 0.15)",
                    name: "Confidence Interval",
                  },
                ]}
                layout={{
                  height: 360,
                  margin: { t: 20, r: 20, b: 40, l: 60 },
                  hovermode: "x unified",
                  template: "plotly_white",
                  yaxis: { title: "Price ($)", tickprefix: "$" },
                  xaxis: { title: "Date" },
                  legend: { orientation: "h", y: -0.2 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
              />

              <div className="prediction-table">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Predicted</th>
                      <th>Lower</th>
                      <th>Upper</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentRows.map((row) => (
                      <tr key={row.date}>
                        <td>{row.date}</td>
                        <td>{`$${Number(row.yhat).toFixed(2)}`}</td>
                        <td>{`$${Number(row.lower).toFixed(2)}`}</td>
                        <td>{`$${Number(row.upper).toFixed(2)}`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">
              No predictions available for the selected asset.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

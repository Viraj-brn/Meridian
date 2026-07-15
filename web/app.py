"""
Flask Web Application — Dashboard for the Index Tracker.

Pure Flask + HTML + CSS (no JavaScript frameworks).
Serves pre-generated matplotlib charts and computed analytics data.
"""

import logging
import sys
from pathlib import Path

from flask import Flask, render_template

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH, INDEX_NAME, CHARTS_DIR
from database.connection import DatabaseManager
from database import queries as Q

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    @app.route("/")
    def dashboard():
        """Main dashboard — overview of index performance and key metrics."""
        data = _load_dashboard_data()
        return render_template("dashboard.html", **data)

    @app.route("/analytics")
    def analytics():
        """Risk analytics page — volatility, VaR, factor decomposition."""
        data = _load_analytics_data()
        return render_template("analytics.html", **data)

    @app.route("/constituents")
    def constituents():
        """Constituent table — all stocks with weights and metrics."""
        data = _load_constituents_data()
        return render_template("constituents.html", **data)

    @app.route("/methodology")
    def methodology():
        """Methodology page — how the index is constructed."""
        return render_template("methodology.html")

    return app


def _load_dashboard_data() -> dict:
    """Load data for the dashboard page."""
    data = {
        "index_name": INDEX_NAME,
        "metrics": {},
        "sector_weights": {},
        "top_holdings": [],
        "has_charts": False,
    }

    try:
        with DatabaseManager(DATABASE_PATH) as db:
            # Index levels
            levels = db.get_index_levels(INDEX_NAME)
            if not levels.empty:
                latest_level = levels["index_level"].iloc[-1]
                first_level = levels["index_level"].iloc[0]
                total_ret = (latest_level / first_level - 1) * 100

                # Recent performance
                if len(levels) > 20:
                    ret_1m = (
                        levels["index_level"].iloc[-1]
                        / levels["index_level"].iloc[-21] - 1
                    ) * 100
                else:
                    ret_1m = 0

                if len(levels) > 252:
                    ret_1y = (
                        levels["index_level"].iloc[-1]
                        / levels["index_level"].iloc[-253] - 1
                    ) * 100
                else:
                    ret_1y = 0

                data["metrics"] = {
                    "current_level": f"{latest_level:,.2f}",
                    "total_return": f"{total_ret:+.2f}%",
                    "return_1m": f"{ret_1m:+.2f}%",
                    "return_1y": f"{ret_1y:+.2f}%",
                    "n_trading_days": len(levels),
                    "start_date": str(levels.index[0].date() if hasattr(levels.index[0], 'date') else levels.index[0]),
                    "end_date": str(levels.index[-1].date() if hasattr(levels.index[-1], 'date') else levels.index[-1]),
                }

            # Top holdings
            top = db.execute_query(
                Q.TOP_N_BY_WEIGHT, (INDEX_NAME, INDEX_NAME, 10)
            )
            if not top.empty:
                data["top_holdings"] = top.to_dict("records")

            # Sector breakdown
            sectors = db.execute_query(
                Q.SECTOR_BREAKDOWN, (INDEX_NAME, INDEX_NAME)
            )
            if not sectors.empty:
                data["sector_weights"] = dict(
                    zip(sectors["sector"], sectors["weight_pct"])
                )

            # DB summary
            summary = db.execute_query(Q.DATABASE_SUMMARY)
            if not summary.empty:
                data["db_summary"] = dict(
                    zip(summary["entity"], summary["count"])
                )

    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}")
        data["error"] = str(e)

    # Check if charts exist
    data["has_charts"] = (CHARTS_DIR / "performance.png").exists()

    return data


def _load_analytics_data() -> dict:
    """Load data for the analytics page."""
    data = {
        "index_name": INDEX_NAME,
        "risk_metrics": {},
        "has_charts": False,
    }

    try:
        with DatabaseManager(DATABASE_PATH) as db:
            levels = db.get_index_levels(INDEX_NAME)
            if not levels.empty:
                returns = levels["daily_return"].dropna()

                import numpy as np
                from analytics.performance import performance_summary

                # Get benchmark returns if available
                benchmark_prices = db.get_prices(["SPY"])
                bench_rets = None
                if not benchmark_prices.empty and "SPY" in benchmark_prices.columns:
                    bench_rets = np.log(
                        benchmark_prices["SPY"]
                        / benchmark_prices["SPY"].shift(1)
                    ).dropna()

                summary = performance_summary(returns, bench_rets)

                # Format for display
                data["risk_metrics"] = {
                    "annualized_return": f"{summary['annualized_return']*100:.2f}%",
                    "annualized_vol": f"{summary['annualized_volatility']*100:.2f}%",
                    "sharpe_ratio": f"{summary['sharpe_ratio']:.3f}",
                    "sortino_ratio": f"{summary['sortino_ratio']:.3f}",
                    "max_drawdown": f"{summary['max_drawdown']*100:.2f}%",
                    "calmar_ratio": f"{summary['calmar_ratio']:.3f}",
                    "var_95": f"{summary['var_95']*100:.2f}%",
                    "cvar_95": f"{summary['cvar_95']*100:.2f}%",
                    "skewness": f"{summary['skewness']:.3f}",
                    "kurtosis": f"{summary['kurtosis']:.3f}",
                    "positive_days": f"{summary['positive_days_pct']*100:.1f}%",
                }

                if "tracking_error" in summary:
                    data["risk_metrics"]["tracking_error"] = f"{summary['tracking_error']*100:.2f}%"
                    data["risk_metrics"]["information_ratio"] = f"{summary['information_ratio']:.3f}"
                    data["risk_metrics"]["active_return"] = f"{summary['active_return_annualized']*100:.2f}%"

    except Exception as e:
        logger.error(f"Error loading analytics data: {e}")
        data["error"] = str(e)

    data["has_charts"] = (CHARTS_DIR / "volatility.png").exists()

    return data


def _load_constituents_data() -> dict:
    """Load data for the constituents page."""
    data = {
        "index_name": INDEX_NAME,
        "constituents": [],
        "n_stocks": 0,
        "n_sectors": 0,
    }

    try:
        with DatabaseManager(DATABASE_PATH) as db:
            # Get latest constituents with weights
            top = db.execute_query(
                Q.TOP_N_BY_WEIGHT, (INDEX_NAME, INDEX_NAME, 100)
            )
            if not top.empty:
                data["constituents"] = top.to_dict("records")
                data["n_stocks"] = len(top)
                data["n_sectors"] = top["sector"].nunique()

            # Anomalies
            anomalies = db.execute_query(Q.PRICE_ANOMALIES)
            if not anomalies.empty:
                data["anomalies"] = anomalies.head(10).to_dict("records")

            # Data gaps
            gaps = db.execute_query(Q.DATA_QUALITY_GAPS)
            if not gaps.empty:
                data["data_gaps"] = gaps.head(10).to_dict("records")

    except Exception as e:
        logger.error(f"Error loading constituents data: {e}")
        data["error"] = str(e)

    return data


if __name__ == "__main__":
    from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

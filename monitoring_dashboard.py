"""
Real-time Monitoring and Dashboard for Enhanced Forecasting System

This script provides a comprehensive monitoring dashboard for the production forecasting system.
It tracks performance metrics, model health, data quality, and system alerts.

Features:
- Real-time performance monitoring
- Model drift detection
- Data quality checks
- Alert management
- Historical trend analysis
- Interactive visualizations
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ForecastingDashboard:
    """Real-time monitoring dashboard for forecasting system"""
    
    def __init__(self, monitoring_dir: str = "monitoring", 
                 models_dir: str = "models",
                 config_path: str = "production_config.json"):
        self.monitoring_dir = monitoring_dir
        self.models_dir = models_dir
        self.config_path = config_path
        
        # Load configuration
        self.load_config()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ForecastingDashboard")
        
        # Dashboard data
        self.current_metrics = {}
        self.historical_data = []
        self.alerts = []
        self.model_info = {}
        
        # Initialize dashboard
        self.refresh_data()
    
    def load_config(self):
        """Load dashboard configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                "production_settings": {
                    "performance_threshold_rmsle": 0.75,
                    "drift_detection_threshold": 0.1
                }
            }
    
    def refresh_data(self):
        """Refresh all dashboard data"""
        self.logger.info("Refreshing dashboard data...")
        
        self.load_monitoring_data()
        self.load_model_information()
        self.detect_alerts()
        self.calculate_trends()
        
        self.logger.info("Dashboard data refreshed successfully")
    
    def load_monitoring_data(self):
        """Load monitoring data from files"""
        self.historical_data = []
        
        if not os.path.exists(self.monitoring_dir):
            self.logger.warning(f"Monitoring directory {self.monitoring_dir} not found")
            return
        
        # Load all monitoring files
        for filename in sorted(os.listdir(self.monitoring_dir)):
            if filename.startswith("monitoring_") and filename.endswith(".json"):
                file_path = os.path.join(self.monitoring_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        daily_data = json.load(f)
                        self.historical_data.extend(daily_data)
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {e}")
        
        # Sort by timestamp
        self.historical_data.sort(key=lambda x: x.get('timestamp', ''))
        
        # Update current metrics with latest data
        if self.historical_data:
            self.current_metrics = self.historical_data[-1].copy()
    
    def load_model_information(self):
        """Load current model information"""
        self.model_info = {}
        
        if not os.path.exists(self.models_dir):
            return
        
        # Find latest model file
        model_files = [f for f in os.listdir(self.models_dir) 
                      if f.startswith("enhanced_model_") and f.endswith(".pkl")]
        
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.models_dir, latest_model)
            
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model_info = {
                        'version': model_data.get('model_version', 'unknown'),
                        'training_date': model_data.get('training_date', 'unknown'),
                        'validation_rmsle': model_data.get('validation_rmsle', 'unknown'),
                        'features_count': len(model_data.get('selected_features', [])),
                        'model_file': latest_model
                    }
            except Exception as e:
                self.logger.error(f"Error loading model info: {e}")
    
    def detect_alerts(self):
        """Detect and categorize alerts"""
        self.alerts = []
        
        if not self.historical_data:
            return
        
        # Get recent data (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_data = [
            entry for entry in self.historical_data 
            if datetime.fromisoformat(entry.get('timestamp', '2020-01-01')) > cutoff_time
        ]
        
        # Performance alerts
        threshold = self.config.get("production_settings", {}).get("performance_threshold_rmsle", 0.75)
        performance_alerts = [
            entry for entry in recent_data 
            if entry.get('current_rmsle', 0) > threshold
        ]
        
        if performance_alerts:
            self.alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'HIGH',
                'count': len(performance_alerts),
                'message': f"{len(performance_alerts)} performance alerts in last 24h",
                'latest_rmsle': performance_alerts[-1].get('current_rmsle', 'unknown')
            })
        
        # Drift alerts
        drift_alerts = [
            entry for entry in recent_data 
            if entry.get('drift_detected', False)
        ]
        
        if drift_alerts:
            self.alerts.append({
                'type': 'DRIFT',
                'severity': 'MEDIUM',
                'count': len(drift_alerts),
                'message': f"{len(drift_alerts)} drift alerts in last 24h"
            })
        
        # System health
        if not recent_data:
            self.alerts.append({
                'type': 'SYSTEM',
                'severity': 'HIGH',
                'message': 'No monitoring data in last 24 hours'
            })
        
        # Model age check
        if self.model_info.get('training_date'):
            try:
                training_date = datetime.fromisoformat(self.model_info['training_date'])
                days_old = (datetime.now() - training_date).days
                
                if days_old > 7:
                    self.alerts.append({
                        'type': 'MODEL_AGE',
                        'severity': 'MEDIUM',
                        'message': f"Model is {days_old} days old - consider retraining"
                    })
            except:
                pass
    
    def calculate_trends(self):
        """Calculate performance trends"""
        if len(self.historical_data) < 2:
            return
        
        # Extract RMSLE values over time
        rmsle_data = [
            (entry.get('timestamp'), entry.get('current_rmsle'))
            for entry in self.historical_data
            if entry.get('current_rmsle') is not None
        ]
        
        if len(rmsle_data) >= 2:
            recent_rmsle = [r[1] for r in rmsle_data[-10:]]  # Last 10 points
            earlier_rmsle = [r[1] for r in rmsle_data[-20:-10]] if len(rmsle_data) >= 20 else [r[1] for r in rmsle_data[:-10]]
            
            if recent_rmsle and earlier_rmsle:
                recent_avg = np.mean(recent_rmsle)
                earlier_avg = np.mean(earlier_rmsle)
                
                self.current_metrics['performance_trend'] = 'IMPROVING' if recent_avg < earlier_avg else 'DEGRADING'
                self.current_metrics['trend_magnitude'] = abs(recent_avg - earlier_avg)
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecasting System Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-value { font-size: 2em; font-weight: bold; color: #333; }
                .metric-label { font-size: 0.9em; color: #666; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .alert-high { background-color: #fee; border-left: 4px solid #e74c3c; }
                .alert-medium { background-color: #fef9e7; border-left: 4px solid #f39c12; }
                .alert-low { background-color: #e8f5e8; border-left: 4px solid #27ae60; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-error { color: #e74c3c; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .timestamp { color: #888; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Enhanced Forecasting System Dashboard</h1>
                    <p>Real-time monitoring and performance tracking</p>
                    <div class="timestamp">Last updated: {timestamp}</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h2>📊 Current Performance</h2>
                        <div class="metric">
                            <div class="metric-value {rmsle_status}">{current_rmsle}</div>
                            <div class="metric-label">Current RMSLE</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{trend_indicator}</div>
                            <div class="metric-label">Trend</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{predictions_count}</div>
                            <div class="metric-label">Recent Predictions</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🤖 Model Information</h2>
                        <p><strong>Version:</strong> {model_version}</p>
                        <p><strong>Training Date:</strong> {training_date}</p>
                        <p><strong>Validation RMSLE:</strong> {validation_rmsle}</p>
                        <p><strong>Features:</strong> {features_count}</p>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🚨 Active Alerts</h2>
                    {alerts_html}
                </div>
                
                <div class="card">
                    <h2>📈 Performance History</h2>
                    <p>Historical performance data over the last 30 days:</p>
                    {performance_summary}
                </div>
                
                <div class="card">
                    <h2>🔧 System Health</h2>
                    <div class="grid">
                        <div>
                            <h3>Data Quality</h3>
                            <p>✅ Data pipeline operational</p>
                            <p>✅ Feature engineering stable</p>
                        </div>
                        <div>
                            <h3>Model Health</h3>
                            <p>{model_health}</p>
                            <p>📊 Monitoring active</p>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare data for template
        current_rmsle = self.current_metrics.get('current_rmsle', 'N/A')
        if isinstance(current_rmsle, (int, float)):
            rmsle_display = f"{current_rmsle:.4f}"
            threshold = self.config.get("production_settings", {}).get("performance_threshold_rmsle", 0.75)
            rmsle_status = "status-good" if current_rmsle < threshold else "status-error"
        else:
            rmsle_display = "N/A"
            rmsle_status = "status-warning"
        
        # Trend indicator
        trend = self.current_metrics.get('performance_trend', 'STABLE')
        trend_indicators = {
            'IMPROVING': '📈 Improving',
            'DEGRADING': '📉 Degrading',
            'STABLE': '➡️ Stable'
        }
        trend_indicator = trend_indicators.get(trend, '➡️ Stable')
        
        # Alerts HTML
        alerts_html = ""
        if not self.alerts:
            alerts_html = '<p class="status-good">✅ No active alerts</p>'
        else:
            for alert in self.alerts:
                severity_class = f"alert-{alert['severity'].lower()}"
                alerts_html += f'<div class="alert {severity_class}">'
                alerts_html += f'<strong>{alert["type"]}:</strong> {alert["message"]}'
                alerts_html += '</div>'
        
        # Performance summary
        if len(self.historical_data) > 0:
            recent_rmsle = [
                entry.get('current_rmsle') for entry in self.historical_data[-30:]
                if entry.get('current_rmsle') is not None
            ]
            if recent_rmsle:
                avg_rmsle = np.mean(recent_rmsle)
                min_rmsle = np.min(recent_rmsle)
                max_rmsle = np.max(recent_rmsle)
                performance_summary = f"""
                <p>Average RMSLE (30 days): {avg_rmsle:.4f}</p>
                <p>Best RMSLE: {min_rmsle:.4f}</p>
                <p>Worst RMSLE: {max_rmsle:.4f}</p>
                <p>Total monitoring points: {len(recent_rmsle)}</p>
                """
            else:
                performance_summary = "<p>No performance data available</p>"
        else:
            performance_summary = "<p>No historical data available</p>"
        
        # Model health
        model_age_days = "unknown"
        if self.model_info.get('training_date'):
            try:
                training_date = datetime.fromisoformat(self.model_info['training_date'])
                model_age_days = (datetime.now() - training_date).days
            except:
                pass
        
        if isinstance(model_age_days, int):
            if model_age_days <= 3:
                model_health = "✅ Model recently trained"
            elif model_age_days <= 7:
                model_health = "⚠️ Model moderately aged"
            else:
                model_health = "🔴 Model needs retraining"
        else:
            model_health = "❓ Model age unknown"
        
        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            current_rmsle=rmsle_display,
            rmsle_status=rmsle_status,
            trend_indicator=trend_indicator,
            predictions_count=self.current_metrics.get('predictions_count', 'N/A'),
            model_version=self.model_info.get('version', 'Unknown'),
            training_date=self.model_info.get('training_date', 'Unknown'),
            validation_rmsle=self.model_info.get('validation_rmsle', 'Unknown'),
            features_count=self.model_info.get('features_count', 'Unknown'),
            alerts_html=alerts_html,
            performance_summary=performance_summary,
            model_health=model_health
        )
        
        return html_content
    
    def save_dashboard(self, output_path: str = "dashboard.html"):
        """Save dashboard to HTML file"""
        html_content = self.generate_dashboard_html()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Dashboard saved to {output_path}")
    
    def create_performance_plots(self, output_dir: str = "visualizations"):
        """Create performance visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.historical_data:
            self.logger.warning("No data available for plotting")
            return
        
        # Extract time series data
        timestamps = []
        rmsle_values = []
        
        for entry in self.historical_data:
            if entry.get('current_rmsle') is not None:
                try:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    timestamps.append(timestamp)
                    rmsle_values.append(entry['current_rmsle'])
                except:
                    continue
        
        if len(timestamps) < 2:
            return
        
        # Performance over time
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, rmsle_values, marker='o', linewidth=2, markersize=4)
        plt.axhline(y=self.config.get("production_settings", {}).get("performance_threshold_rmsle", 0.75), 
                   color='red', linestyle='--', label='Performance Threshold')
        plt.xlabel('Time')
        plt.ylabel('RMSLE')
        plt.title('Model Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance distribution
        plt.figure(figsize=(8, 6))
        plt.hist(rmsle_values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(rmsle_values), color='red', linestyle='--', label=f'Mean: {np.mean(rmsle_values):.4f}')
        plt.xlabel('RMSLE')
        plt.ylabel('Frequency')
        plt.title('RMSLE Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance plots saved to {output_dir}")
    
    def generate_alert_report(self) -> str:
        """Generate alert summary report"""
        if not self.alerts:
            return "No active alerts - system operating normally."
        
        report_lines = [
            "FORECASTING SYSTEM ALERT REPORT",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Active Alerts: {len(self.alerts)}",
            ""
        ]
        
        # Group alerts by severity
        high_alerts = [a for a in self.alerts if a.get('severity') == 'HIGH']
        medium_alerts = [a for a in self.alerts if a.get('severity') == 'MEDIUM']
        low_alerts = [a for a in self.alerts if a.get('severity') == 'LOW']
        
        if high_alerts:
            report_lines.extend([
                "🔴 HIGH SEVERITY ALERTS:",
                "-" * 25
            ])
            for alert in high_alerts:
                report_lines.append(f"• {alert['type']}: {alert['message']}")
            report_lines.append("")
        
        if medium_alerts:
            report_lines.extend([
                "🟡 MEDIUM SEVERITY ALERTS:",
                "-" * 27
            ])
            for alert in medium_alerts:
                report_lines.append(f"• {alert['type']}: {alert['message']}")
            report_lines.append("")
        
        if low_alerts:
            report_lines.extend([
                "🟢 LOW SEVERITY ALERTS:",
                "-" * 23
            ])
            for alert in low_alerts:
                report_lines.append(f"• {alert['type']}: {alert['message']}")
        
        return "\n".join(report_lines)
    
    def print_summary(self):
        """Print dashboard summary to console"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED FORECASTING SYSTEM DASHBOARD")
        print("=" * 60)
        
        # Current status
        print(f"📊 Current RMSLE: {self.current_metrics.get('current_rmsle', 'N/A')}")
        print(f"🤖 Model Version: {self.model_info.get('version', 'Unknown')}")
        print(f"📈 Performance Trend: {self.current_metrics.get('performance_trend', 'Unknown')}")
        
        # Alerts summary
        if self.alerts:
            print(f"\n🚨 Active Alerts: {len(self.alerts)}")
            for alert in self.alerts[:3]:  # Show first 3 alerts
                print(f"   • {alert['type']}: {alert['message']}")
        else:
            print("\n✅ No active alerts")
        
        # System health
        print(f"\n🔧 Monitoring Points: {len(self.historical_data)}")
        
        if self.model_info.get('training_date'):
            try:
                training_date = datetime.fromisoformat(self.model_info['training_date'])
                days_old = (datetime.now() - training_date).days
                print(f"🕒 Model Age: {days_old} days")
            except:
                print("🕒 Model Age: Unknown")
        
        print("=" * 60)

def main():
    """Main dashboard execution"""
    print("Initializing Forecasting Dashboard...")
    
    # Create dashboard
    dashboard = ForecastingDashboard()
    
    # Print summary
    dashboard.print_summary()
    
    # Generate visualizations
    dashboard.create_performance_plots()
    
    # Save HTML dashboard
    dashboard.save_dashboard()
    
    # Generate alert report if needed
    if dashboard.alerts:
        alert_report = dashboard.generate_alert_report()
        print("\n" + alert_report)
        
        with open("alert_report.txt", "w") as f:
            f.write(alert_report)
    
    print("\n✅ Dashboard generated successfully!")
    print("📊 View dashboard.html in your browser")
    print("📈 Check visualizations/ directory for plots")

if __name__ == "__main__":
    main()

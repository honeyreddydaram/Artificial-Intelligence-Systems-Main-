import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import os
import pandas as pd

METRICS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'deployment_metrics.csv')
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'user_feedback.csv')

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = []
            # Inference time metrics
            if os.path.exists(METRICS_FILE):
                metrics_df = pd.read_csv(METRICS_FILE)
                if 'response_time' in metrics_df.columns:
                    avg_response = metrics_df['response_time'].mean()
                    metrics.append(f'inference_response_time_mean {avg_response}')
                    metrics.append(f'inference_response_time_count {len(metrics_df)}')
            # User feedback metrics
            if os.path.exists(FEEDBACK_FILE):
                feedback_df = pd.read_csv(FEEDBACK_FILE)
                if 'feedback' in feedback_df.columns:
                    positive = (feedback_df['feedback'] == 'positive').sum()
                    negative = (feedback_df['feedback'] == 'negative').sum()
                    metrics.append(f'user_feedback_positive {positive}')
                    metrics.append(f'user_feedback_negative {negative}')
                    metrics.append(f'user_feedback_total {len(feedback_df)}')
            metrics_output = '\n'.join(metrics) + '\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics_output.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def start_metrics_server(port=8000):
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Prometheus metrics exporter running on port {port}")
    return server

if __name__ == "__main__":
    start_metrics_server()
    while True:
        time.sleep(60)
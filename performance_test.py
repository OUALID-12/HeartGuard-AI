import time
import requests
import json
import statistics

# Configuration
BASE_URL = "http://127.0.0.1:8000"  # Adjust as necessary
ENDPOINTS = [
    {"name": "Home Page", "path": "/", "method": "GET"},
    {"name": "History (Timeline)", "path": "/history/", "method": "GET"},
    {"name": "Predict Page", "path": "/predict/", "method": "GET"},
]

def run_performance_test(iterations=5):
    print(f"Starting Performance Test for HeartGuard AI (Iterations: {iterations})...")
    print("-" * 60)
    
    results = {}

    for endpoint in ENDPOINTS:
        print(f"Testing {endpoint['name']} ({endpoint['path']})...")
        latencies = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = requests.request(endpoint['method'], f"{BASE_URL}{endpoint['path']}")
                latency = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
                else:
                    print(f"  Iteration {i+1}: Failed (Status: {response.status_code})")
            except Exception as e:
                print(f"  Iteration {i+1}: Error ({str(e)})")
        
        if latencies:
            results[endpoint['name']] = {
                "min": min(latencies),
                "max": max(latencies),
                "avg": statistics.mean(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
            print(f"  Avg Latency: {results[endpoint['name']]['avg']:.2f} ms")
        else:
            print(f"  No successful iterations for {endpoint['name']}")

    print("-" * 60)
    print("Performance Summary (ms):")
    print(f"{'Endpoint':<20} | {'Avg':<10} | {'Min':<10} | {'Max':<10}")
    for name, stats in results.items():
        print(f"{name:<20} | {stats['avg']:<10.2f} | {stats['min']:<10.2f} | {stats['max']:<10.2f}")

if __name__ == "__main__":
    # Note: This script assumes the server is running.
    # If using local testing without session, it might hit login redirect.
    run_performance_test()

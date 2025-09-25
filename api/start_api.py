"""
Simple script to start the CyberPuppy API for testing.
"""

import sys
from pathlib import Path
import logging

# Add the project root to the path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    import uvicorn
    from app import app

    port = 8001  # Use different port to avoid conflicts
    print("Starting CyberPuppy API...")
    print(f"API will be available at: http://localhost:{port}")
    print(f"API documentation: http://localhost:{port}/docs")
    print(f"Health check: http://localhost:{port}/healthz")
    print(f"Metrics: http://localhost:{port}/metrics")
    print("\nPress Ctrl+C to stop the server")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)

# -*- coding: utf-8 -*-
"""
run_mcp_server.py - Main entry point to run the QuantumSim MCP server.
"""

import logging
# import sys

# Add the src directory to the Python path
# This allows us to import the quantum_sim package
# sys.path.append('src')

from quantum_sim.mcp_tools import mcp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Starts the MCP server."""
    logger.info('Starting QuantumSim MCP server...')
    mcp.run('streamable-http')

if __name__ == "__main__":
    main()

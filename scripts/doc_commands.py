import sys
import os
import inspect

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.handlers import HANDLERS

def generate_command_docs():
    output_path = "docs/API_COMMANDS.md"
    
    with open(output_path, "w") as f:
        f.write("# API Commands Reference\n\n")
        f.write("This document is auto-generated. It lists all available WebSocket commands grouped by scope.\n\n")
        
        for scope, handlers in HANDLERS.items():
            f.write(f"## Scope: `{scope}`\n\n")
            f.write("| Command | Description |\n")
            f.write("|---------|-------------|\n")
            
            for command, handler in handlers.items():
                doc = inspect.getdoc(handler)
                if doc:
                    # Take first line of docstring
                    description = doc.split('\n')[0]
                else:
                    description = "No description available."
                
                f.write(f"| `{command}` | {description} |\n")
            
            f.write("\n")
            
    print(f"✅ Documentation generated at {output_path}")

if __name__ == "__main__":
    generate_command_docs()

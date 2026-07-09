from pathlib import Path
import os
from service.api import create_app

workspace = Path(os.environ.get("ARKIT_WORKSPACE", "/tmp/arkit_workspace"))
app = create_app(workspace)

from app.app import app

import os
with open("debug_log.txt", "w") as f:
    f.write("🔥 실행된 app.py 위치: " + os.path.abspath(__file__) + "\n")

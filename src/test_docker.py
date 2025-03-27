# Created by MacBook Pro at 27.03.25


import debugpy
import time
import pydevd_pycharm


time.sleep(3)
pydevd_pycharm.settrace(
    '130.83.185.158',  # 🔥 very important: NOT your Mac IP
    port=5678,
    stdoutToServer=True,
    stderrToServer=True,
    suspend=True
)

# debugpy.listen(("0.0.0.0", 5678))
a = 20
print("✅ Waiting for debugger to attach on port 5678...")
debugpy.wait_for_client()
print("test docker")

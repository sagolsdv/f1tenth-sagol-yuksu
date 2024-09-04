#!/usr/bin/env python3

import os
import subprocess

output = subprocess.check_output(("sudo", "/usr/bin/docker", "ps"))
for line in output.decode("utf-8").split("\n")[1:]:
    if line.strip() == "":
        continue
    if line.find("sagol:base") < 0:
        continue
    image_id = line.split(" ")[0]
    os.system(f"sudo docker exec -it {image_id} /bin/bash")
    break

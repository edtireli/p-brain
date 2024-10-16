import sys
import subprocess

ids = sys.argv[1:]

command_template = "python3 main.py --id {} --option 6"

for id in ids:
    command = command_template.format(id)
    print(f"Running: {command}") 
    subprocess.run(command, shell=True)

import yaml
import sys
import random

# Read YAML data from stdin
input_yaml = sys.stdin.read()

# Parse YAML
data = yaml.safe_load(input_yaml)
data['data_path'] = sys.argv[1]
data['save'] = sys.argv[2]
data['load'] = sys.argv[2]
data['master_port'] = 29500 + random.randint(1, 10000) 

print(yaml.dump(data, default_flow_style=True, default_style='"'))

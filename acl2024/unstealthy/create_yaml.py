import yaml
import sys

# Read YAML data from stdin
input_yaml = sys.stdin.read()

# Parse YAML
data = yaml.safe_load(input_yaml)
data['data_path'] = sys.argv[1]
data['save_path'] = sys.argv[2]
data['load_path'] = sys.argv[2]

print(yaml.dump(data, default_flow_style=True, default_style='"'))

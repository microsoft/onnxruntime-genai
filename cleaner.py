import json

filename = 'cmake/presets/CMakeLinuxX64ConfigPresets.json'
# Load the JSON file
with open(filename, 'r') as file:
    data = json.load(file)

# Iterate over the configurePresets list
data['configurePresets'] = [preset for preset in data['configurePresets'] if
                            'clang' not in preset['name'].lower()]

# Write the modified JSON back to the file
with open(filename, 'w') as file:
    json.dump(data, file, indent=2)

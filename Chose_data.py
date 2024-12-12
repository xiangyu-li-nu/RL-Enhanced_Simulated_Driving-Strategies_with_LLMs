import json

# Set the threshold
threshold = 5

# Read the original JSON data
with open('train_new.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter the data above the threshold
filtered_data = [entry for entry in data if entry['reward'] > threshold]

# Save the filtered data to a new JSON file
with open('filtered_train_new.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"The filtered data has been saved to 'filtered_train_new.json'")

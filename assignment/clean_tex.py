"""
Clean the duplicated final.tex file
"""
import re

# Read the corrupted file
with open('d:/Machine Learning/assignment/final.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Function to remove consecutive duplicates from a line
def remove_line_duplicates(line):
    # Remove exact duplicates appearing back-to-back
    while True:
        original = line
        # Try different split points to find duplications
        for i in range(1, len(line)):
            part1 = line[:i]
            part2 = line[i:i+len(part1)]
            if part1 and part1 == part2:
                line = part1 + line[i+len(part1):]
                break
        if line == original:
            break
    return line

# Process line by line
lines = content.split('\n')
cleaned_lines = []

for line in lines:
    cleaned_line = remove_line_duplicates(line)
    cleaned_lines.append(cleaned_line)

# Join back
cleaned_content = '\n'.join(cleaned_lines)

# Remove results/ from image paths
cleaned_content = cleaned_content.replace('{results/', '{')
cleaned_content = cleaned_content.replace('results/','')

# Clean up excessive blank lines
cleaned_content = re.sub(r'\n{4,}', '\n\n\n', cleaned_content)

# Write the cleaned file
with open('d:/Machine Learning/assignment/final.tex', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("File cleaned successfully!")
print("Original file has been cleaned and updated.")

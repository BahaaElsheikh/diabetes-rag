import json

with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f'Total chunks: {len(chunks)}')
print('Valid JSON: Yes')
print()

with_section = sum(1 for c in chunks if c.get('section_number'))
without_section = sum(1 for c in chunks if not c.get('section_number'))
print(f'Chunks WITH section_number: {with_section}')
print(f'Chunks WITHOUT section_number (null): {without_section}')
print()

for i, c in enumerate(chunks[:3]):
    sn = c.get('section_number')
    pn = c.get('page_number')
    txt = c['text'][:100]
    print(f'--- Sample {i+1} ---')
    print(f'  section_number: {sn}')
    print(f'  page_number: {pn}')
    print(f'  text (first 100 chars): {txt}')
    print()

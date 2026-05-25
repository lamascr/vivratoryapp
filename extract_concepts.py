from bs4 import BeautifulSoup

def extract_headings_and_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    with open('concepts.txt', 'a', encoding='utf-8') as out:
        out.write(f"\n--- {file_path} ---\n")
        # Find all headings
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            out.write(f"{header.name}: {header.get_text(strip=True)}\n")
            # Try to get the next paragraph or list
            nxt = header.find_next_sibling(['p', 'ul'])
            if nxt:
                out.write(f"  Content: {nxt.get_text(strip=True)[:500]}...\n")

with open('concepts.txt', 'w', encoding='utf-8') as f:
    f.write('Extracted concepts:\n')

extract_headings_and_content('c:/Carlos/UPV/0.7 Diagnostico Mediante Analizis de Ruido/Tema05_Analisis_TiempoFrecuenciaNoWidgets.html')
extract_headings_and_content('c:/Carlos/UPV/0.7 Diagnostico Mediante Analizis de Ruido/Tema06_FuentesNoWidgets.html')

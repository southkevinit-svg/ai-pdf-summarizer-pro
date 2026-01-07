# 1. Installation (Kostenlose KI-Modelle)
!pip install PyPDF2 transformers torch-sentencepiece

import PyPDF2
from transformers import pipeline
from google.colab import files

# 2. Datei-Upload
print("Bitte lade dein PDF hoch:")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# 3. Text extrahieren
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf(file_name)

# 4. Kostenlose Zusammenfassung (Nutzt das 'BART' Modell von Meta/Facebook)
print("\nKI wird geladen (das dauert beim ersten Mal ca. 30 Sek.)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Den Text in Stücke teilen, damit die KI nicht überlastet wird
summary = summarizer(pdf_text[:1024], max_length=150, min_length=40, do_sample=False)

print("\n--- KOSTENLOSE KI-ZUSAMMENFASSUNG

#pip install pdfplumber
#pip install pypdf
#pip install PyPDF2


import PyPDF2
def pdf_to_text(uploaded_file):

    pdfReader = PyPDF2.PdfReader(uploaded_file)
    no_pages = len(pdfReader.pages)
    text=""
    for page_index in range(no_pages):
        page = pdfReader.pages[page_index]
        text=text+page.extract_text()
    return text



import pdfplumber
def pdf_to_text_plumber(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return '\n'.join(filter(None, text))


from langchain_community.document_loaders import PyPDFLoader
def pdfloader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text = "".join([page.page_content for page in pages])
    return text





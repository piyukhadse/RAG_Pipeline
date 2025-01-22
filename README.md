Multilingual PDF RAG System with Jina and XGLM


Background: Our organization needs to develop a Retrieval-Augmented Generation (RAG) system capable of processing multilingual PDFs, extracting information, and providing summaries and answers to questions based on the content. The system should handle various languages including Urdhu, English, Bengali, and Chinese, and be able to process both scanned and digital PDFs

Objective:  Develop a RAG pipeline for summarizing content and answering questions based on the input PDFs. The system should be scalable to handle large amounts of data (up to 1TB) and provide accurate, relevant responses.




Technical Documentation:
A. Text extraction technique using OCR from PDF: 
1. Setup and Library Import
	Imports necessary libraries:
	⦁	os and Pathlib: For file and directory operations.
	⦁	pdf2image: Converts PDF files into images.
	⦁	pytesseract: OCR library for extracting text from images.
	⦁	json: To save the extracted text to a JSON file.
	⦁	Sets the path to the Tesseract executable using pytesseract.pytesseract.tesseract_cmd.
2. To implement text extraction techniques, including OCR for scanned documents and standard extraction for digital PDFs, use Python libraries Tesseract OCR for scanned documents.
	First, ensure that Tesseract is installed. Install it on your system or via a Python wrapper:
	⦁	For window: https://github.com/tesseract-ocr/tesseract follow to install
	⦁	setting : pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
	⦁	To enable multilingual OCR in Tesseract, install additional language packs:
	⦁	https://github.com/tesseract-ocr/tessdata : [chi_sim.traineddata, ben.traineddata, urd.traineddata]
	⦁	languages = urd, ben, eng, chi_sim 
	⦁	extract text from different pdf in different folders
	⦁	stored extracted text in JSON format 
		
B. Text preprocessing:
	⦁	By using  "re" clean text by removing unwanted characters and patterns:
			"\n, \\, ||, [U+200E], ---, and extra spaces" 
C. Text Chunking:
	⦁	Divide the extracted text into smaller chunks (e.g., paragraphs or sentences) for better retrieval and processing.
	⦁	This chunking ensures that relevant pieces of text can be retrieved to answer questions.
D. Embedding the Text:
Setup and Library Import
	⦁	Imports necessary libraries:
	⦁	XGLMForCausalLM: A pre-trained multilingual causal language model for embedding.
	⦁	generation_tokenizer: Tokenizer corresponding to the model.
	⦁	Document: Represents individual text and its embedding.
	⦁	DocumentArray: Container for multiple Document objects.
	⦁	Flow: A mechanism to create and manage pipelines for indexing, querying, or processing data.
	⦁	Jina Executor: For example, DocCache for caching documents during indexing.
	⦁	PyTorch: Required for running the transformer model and handling tensors.
	⦁	NumPy: For converting PyTorch tensors into NumPy arrays.
			Converted the extracted content into a form that a model can work with. This is typically done by transforming the text into embeddings (numerical vectors) 
   			using a pre-trained language model such as : XGLM-564M is a multilingual autoregressive language model (with 564 million parameters)
		⦁	XGLMTokenizer.from_pretrained("facebook/xglm-564M")
		⦁	XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
					ref: https://huggingface.co/facebook/xglm-564M
					        https://huggingface.co/jinaai/jina-embeddings-v3
E. Store Embeddings in Vector Database using Jina:
	Jina provides an easy way to index and store embeddings.
	"flow = Flow().add(uses="jinahub://Indexers.vectorstore")"
F. Query Processing:
	For answering questions, we first need to embed the user’s query and retrieve relevant text chunks. 
G. Answer Generation with XGLM-2:
	Once retrieve relevant text chunks, you can feed them into XGLM for answer generation.
	⦁	XGLMTokenizer.from_pretrained("facebook/xglm-564M")
	⦁	XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
				ref: https://huggingface.co/facebook/xglm-564M
				        https://huggingface.co/jinaai/jina-embeddings-v3
H. RAG Pipeline for Summarizing and Answering Questions:
	RAG pipeline to summarize content and answer questions based on the extracted PDFs.

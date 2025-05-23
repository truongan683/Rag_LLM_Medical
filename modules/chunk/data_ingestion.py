# import os
# from concurrent.futures import ThreadPoolExecutor
# from chunk.document_processor import process_document
# from chunk.chromadb_manager import save_to_chromadb
# from chunk.embeddings import embed_texts, tokenizer_embedding, model_embedding
# from config import FOLDER_PATH, chunk_logger

# def ingest_new_data(source_path=None, text_list=None):
#     """
#     Ingest new data from a folder of text files or a list of texts.
    
#     Args:
#         source_path (str, optional): Path to folder containing .txt files.
#         text_list (list, optional): List of text strings to process.
    
#     Returns:
#         list: List of processed document chunks.
#     """
#     documents = []
#     counter = 0

#     if source_path:
#         files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith(".txt")]
#         total_files = len(files)
#         if total_files == 0:
#             chunk_logger.error(f"No .txt files found in {source_path}!")
#             return []
        
#         chunk_logger.info(f"Found {total_files} .txt files to process in {source_path}")
        
#         with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#             futures = [
#                 executor.submit(process_document, file_path, i, total_files, counter + i * 1000, tokenizer_embedding, model_embedding)
#                 for i, file_path in enumerate(files)
#             ]
#             for future in futures:
#                 try:
#                     chunk_data, new_counter = future.result()
#                     documents.extend(chunk_data)
#                     counter = max(counter, new_counter)
#                 except Exception as e:
#                     chunk_logger.error(f"Error processing file: {str(e)}")
    
#     elif text_list:
#         chunk_logger.info(f"Processing {len(text_list)} text inputs")
#         for i, text in enumerate(text_list):
#             temp_file = f"temp_input_{i}.txt"
#             try:
#                 with open(temp_file, "w", encoding="utf-8") as f:
#                     f.write(text)
                
#                 chunk_data, new_counter = process_document(temp_file, i, len(text_list), counter, tokenizer_embedding, model_embedding)
#                 documents.extend(chunk_data)
#                 counter = max(counter, new_counter)
#             except Exception as e:
#                 chunk_logger.error(f"Error processing text input {i}: {str(e)}")
#             finally:
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
    
#     else:
#         chunk_logger.error("No source path or text list provided!")
#         return []
    
#     if not documents:
#         chunk_logger.warning("No chunks generated from input data!")
#         return []
    
#     texts = [doc["text"] for doc in processed_docs]
#     embeddings = embed_texts(texts, batch_size=32)
#     save_to_chromadb(documents, embeddings)
#     chunk_logger.info(f"Successfully ingested {len(documents)} chunks")
#     return documents

# def ingest_from_default_folder():
#     """Ingest data from the default folder specified in config."""
#     return ingest_new_data(source_path=FOLDER_PATH)
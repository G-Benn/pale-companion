# %% [markdown]
# Following the steps [here](https://haystack.deepset.ai/tutorials/07_rag_generator)

# %%
# Import some files
import pickle
with open('../data/chapter_fmt_list.pkl','rb') as f:
    chapters = pickle.load(f)

# %%
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


# %%
from haystack import Document
chapter_documents = [Document.from_dict(d) for d in chapters]

# %%
# Try with a few different Preprocessors on either passages (too long) or sentence. We'll probably need to tinker with this
from haystack.nodes import PreProcessor

word_preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
    split_overlap=20,
    progress_bar=True, 
    add_page_number=True
)

sentence_preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=10,
    split_respect_sentence_boundary=False,
    split_overlap=2,
    progress_bar=True, 
    add_page_number=True
)
# Should add max_chars_check or similar once we get to the point we do a dense retreival model

# %%
docs_word = word_preprocessor.process(chapter_documents)
docs_sentence = sentence_preprocessor.process(chapter_documents)
len(docs_word), len(docs_sentence)

# %%
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever


# Initialize FAISS document store.
# Set `return_embedding` to `True`, so generator doesn't have to perform re-embedding
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

# Initialize DPR Retriever to encode documents, encode question and query documents
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=False,
)

# Initialize RAG Generator
generator = RAGenerator(
    model_name_or_path="facebook/rag-token-nq",
    use_gpu=True,
    top_k=1,
    max_length=200,
    min_length=2,
    embed_title=False,
    num_beams=2,
)


# %%
# Delete existing documents in documents store
document_store.delete_documents()

# Batch the write / embeddings update because my machine doesn't have much memory
# chunks = [docs_word[x:x+100] for x in range(0, len(docs_word), 100)]

# Write documents to document store
document_store.write_documents(docs_word)

# Add documents embeddings to index
document_store.update_embeddings(retriever=retriever)


# %%
# document_store.save("gen_qa_v1.faiss")
# new_document_store = FAISSDocumentStore.load("my_faiss_index.faiss")

# %%
QUESTIONS = [
    "Who is Avery?",
    "Where does Verona live?",
    "What is Toadswallow?",
    "How many Carmine conspirators are there?",
    "What does Alpeona do?",
    "Why is Gilkey ostracized?",
    "What is Snowdrop?",
    "Why is Maricaca scared of?",
    "What are the opinions of Charles?",
    "How many others live in Kennet?",
    "Why was Seth Forsaken?",
    "Is Alexander dead?",
    "Where is Bristow?",
    "What is the Wolf?",
    "What is the Red Heron?",
    "Can we talk about the girls?",
    "Who has a crush on Avery?",
    "What are the most powerful Others?",
    "What are the boons for Sootsleeves path?"
]


# %%
# Or alternatively use the Pipeline class
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 5}})
    print_answers(res, details="minimum")


# %%




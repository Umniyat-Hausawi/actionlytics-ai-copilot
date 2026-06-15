import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

sys.path.append(os.path.dirname(__file__))

# ===== Load Knowledge Base from external file =====
def _load_knowledge_base():
    kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.md')
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            return f.read()
    raise FileNotFoundError(f'knowledge_base.md not found at {kb_path}')


# ===== Extract metadata from chunk text =====
def _extract_metadata(text):
    """
    Reads the <!-- category: X | source: Y | market: Z --> comment
    and returns a metadata dict.
    """
    metadata = {'category': 'general', 'source': 'unknown', 'market': 'global'}
    if '<!-- category:' in text:
        try:
            comment = text.split('<!--')[1].split('-->')[0]
            for part in comment.split('|'):
                key, _, value = part.strip().partition(':')
                metadata[key.strip()] = value.strip()
        except Exception:
            pass
    return metadata


# ===== Build RAG =====
def build_rag():
    try:
        knowledge_base = _load_knowledge_base()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        raw_chunks = splitter.create_documents([knowledge_base])

        # Attach metadata to each chunk
        chunks = []
        for chunk in raw_chunks:
            metadata = _extract_metadata(chunk.page_content)
            chunks.append(Document(
                page_content=chunk.page_content,
                metadata=metadata
            ))

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    except Exception as e:
        raise RuntimeError(f'Failed to build RAG: {e}')


# ===== Query RAG with similarity threshold =====
SIMILARITY_THRESHOLD = 0.75

def query_rag(vectorstore, question, k=3):
    """
    Returns structured context only if similarity is above threshold.
    If no relevant benchmark found, returns empty string — chatbot won't hallucinate.
    """
    try:
        results = vectorstore.similarity_search_with_score(question, k=k)

        relevant = []
        for doc, score in results:
            similarity = 1 / (1 + score)
            if similarity >= SIMILARITY_THRESHOLD:
                relevant.append((doc, similarity))

        if not relevant:
            return ''

        context_parts = []
        for doc, similarity in relevant:
            meta     = doc.metadata
            category = meta.get('category', 'general')
            source   = meta.get('source', 'unknown')
            market   = meta.get('market', 'global')
            context_parts.append(
                f"[Benchmark — {category} | Source: {source} | Market: {market}]\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        return ''


# ===== Retrieval Evaluation =====
def evaluate_retrieval(vectorstore):
    """
    Runs 10 test questions and checks if the retrieved chunks
    are relevant to the expected category.
    """
    test_cases = [
        ("What is a good cancellation rate?",           "cancellation_rate"),
        ("معدل الإلغاء الجيد كم يكون؟",                "cancellation_rate"),
        ("What is a good conversion rate for electronics?", "conversion_rate"),
        ("معدل التحويل للإلكترونيات",                  "conversion_rate"),
        ("Best platform for campaigns in Saudi Arabia", "campaign_roi"),
        ("أفضل منصة إعلانات في السعودية",              "campaign_roi"),
        ("What is a good repeat purchase rate?",        "repeat_purchase"),
        ("معدل الشراء المتكرر الجيد",                  "repeat_purchase"),
        ("Electronics average order value benchmark",   "electronics_specific"),
        ("How to reduce cart abandonment?",             "cart_abandonment"),
    ]

    passed  = 0
    results = []

    for question, expected_category in test_cases:
        raw = vectorstore.similarity_search_with_score(question, k=1)
        if raw:
            doc, score      = raw[0]
            found_category  = doc.metadata.get('category', 'unknown')
            similarity      = round(1 / (1 + score), 3)
            is_correct      = found_category == expected_category
            if is_correct:
                passed += 1
            results.append({
                'question':          question,
                'expected_category': expected_category,
                'found_category':    found_category,
                'similarity':        similarity,
                'passed':            is_correct
            })

    accuracy = round(passed / len(test_cases) * 100, 1)
    return results, accuracy


# ===== Main =====
if __name__ == '__main__':
    print('Building RAG knowledge base...')
    try:
        vectorstore = build_rag()
        print('RAG built successfully!')

        print('\n===== Retrieval Evaluation =====')
        results, accuracy = evaluate_retrieval(vectorstore)
        for r in results:
            status = '✅' if r['passed'] else '❌'
            print(f"{status} [{r['similarity']}] {r['question'][:50]}")
            if not r['passed']:
                print(f"   Expected: {r['expected_category']} | Found: {r['found_category']}")
        print(f'\nRetrieval Accuracy: {accuracy}%')

        print('\n===== Sample Queries =====')
        test_questions = [
            "What is a good cancellation rate?",
            "معدل الإلغاء الجيد كم يكون؟",
            "Best campaigns for electronics store",
            "Saudi Arabia ecommerce market size",
            "What is a random question that has no benchmark?",
        ]

        for q in test_questions:
            print(f'\nQuestion: {q}')
            result = query_rag(vectorstore, q)
            if result:
                print(f'Result: {result[:200]}...')
            else:
                print('Result: No relevant benchmark found — RAG not used')
            print('-' * 50)

    except Exception as e:
        print(f'RAG Engine failed: {e}')
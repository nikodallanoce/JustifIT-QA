import argparse
import random
from functools import partial

import pandas as pd

import datasets
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import TextGenerationPipeline

from Qwen3Reranker import Qwen3Reranker
from build_vector_db import LegalFAISSRetriever, BuildConfig
from feedback_loop import (
    get_article_text_retrieved_or_labelled,
    prepare_llm_input,
    generate_text,
    clean_llm_output_content,
)

gen_evidence_prompt = """RUOLO
Sei un assistente legale specializzato nell’estrazione di evidenze testuali. Il tuo compito è individuare e copiare VERBATIM (testo identico) tutte le porzioni degli articoli forniti che sono rilevanti per supportare, giustificare o verificare la RISPOSTA data alla DOMANDA.

OBIETTIVO
Dato:
1) una DOMANDA (ambito legale),
2) una RISPOSTA alla domanda,
3) una lista JSON di ARTICOLI nel formato:
   [
     {{"doc_id": 1, "content": "..."}} ,
     {{"doc_id": 2, "content": "..."}} ,
     ...
   ]
Devi estrarre dagli ARTICOLI TUTTE le sezioni di testo rilevanti per la risposta. Gli articoli forniti in input provengono da un sistema di retrieval (retriever). Di conseguenza, NON è garantito che tutti gli articoli siano effettivamente rilevanti per la domanda o per la risposta: alcuni possono essere rumore. Devi quindi selezionare solo gli articoli e le porzioni di testo realmente pertinenti, ignorando quelli non rilevanti.

DEFINIZIONE DI “SEZIONE RILEVANTE”
Una sezione è rilevante se:
- contiene una regola, condizione, eccezione, definizione, obbligo, divieto, procedura, requisito, sanzione o criterio interpretativo
  che (i) supporta direttamente una parte della RISPOSTA, oppure (ii) la limita/qualifica (eccezioni, casi particolari), oppure (iii) è necessaria per interpretarla correttamente.
- include anche frasi che definiscono termini o rinvii (es. “ai sensi di…”, “in deroga a…”) se utili alla comprensione della risposta.

REGOLE CRITICHE (OBBLIGATORIE)
1) VERBATIM assoluto: copia il testo esattamente come appare in "content".
   - NON usare "..." o altre ellissi.
   - NON parafrasare, NON riassumere, NON correggere refusi.
2) COMPLETEZZA: se una disposizione è rilevante, includi l’intera frase/periodo necessario a mantenere il senso giuridico.
   - Se il significato dipende da più frasi contigue, includile tutte.
3) GRANULARITÀ: estrai “sezioni” come blocchi di testo coerenti (una o più frasi contigue).
   - Evita estratti troppo corti che perdono contesto.
4) DEDUPLICAZIONE: non ripetere lo stesso identico estratto due volte nello stesso articolo.
5) COPERTURA MULTI-ARTICOLO: se la risposta è supportata da più articoli, includi estratti per ciascuno di essi.
6) SOLO DA INPUT: usa esclusivamente il testo presente negli articoli forniti. Nessuna conoscenza esterna.

FORMATO DI OUTPUT (UNICO OUTPUT CONSENTITO)
Restituisci SOLO un JSON valido (niente testo extra, niente markdown), nel formato:
[
  {{
    "doc_id": <numero>,
    "citations": [
      "<estratto verbatim 1>",
      "<estratto verbatim 2>",
      ...
    ]
  }},
  ...
]

VINCOLI SULL’OUTPUT
- Includi un elemento nella lista di output SOLO per gli articoli da cui hai estratto almeno una citazione.
- "citations" deve essere una lista di stringhe; ogni stringa è un blocco verbatim.
- Mantieni gli a-capo esattamente come nel testo se presenti; se non sei sicuro, mantieni la formattazione originale il più possibile.
- Se non trovi alcuna sezione rilevante in tutti gli articoli, restituisci [].

INPUT
DOMANDA:
{0}

RISPOSTA:
{1}

ARTICOLI (JSON):
{2}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate evidence spans and retrieved article ids for generated answers."
    )

    parser.add_argument(
        "--conversation_ds_path",
        type=str,
        required=True,
        help="Path of the conversation dataset saved on disk."
    )
    parser.add_argument(
        "--llm_model_repo",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF repository of the LLM used to generate evidence."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".",
        help="Cache directory for models/tokenizers/embeddings."
    )
    parser.add_argument(
        "--embedder_model_name",
        type=str,
        default="justifit/DARv2",
        help="Embedding model name/path."
    )
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        help="Enable thinking mode in the tokenizer chat template."
    )
    parser.add_argument(
        "--cutoff_k",
        type=int,
        default=20,
        help="Top-k documents retrieved before optional reranking."
    )
    parser.add_argument(
        "--reranker_cutoff_k",
        type=int,
        default=4,
        help="Top-k documents kept after reranking."
    )
    parser.add_argument(
        "--reranker_bsz",
        type=int,
        default=4,
        help="Reranker batch size."
    )
    parser.add_argument(
        "--retriever_bsz",
        type=int,
        default=16,
        help="Retriever batch size."
    )
    parser.add_argument(
        "--reranker_model_repo",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="HF repository of the reranker model."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/gen_evidence",
        help="Output path where the final dataset will be saved."
    )

    return parser.parse_args()


def generate_evidence(
        questions: str,
        relevant_texts: list[str],
        answers: list[str],
        pipeline: TextGenerationPipeline,
        retriever: LegalFAISSRetriever,
        prompt_in: str,
        cutoff_k: int,
        answer_start_sentence: str,
        retrieve_docs: bool = True,
        reranker: Qwen3Reranker = None,
        reranker_cutoff_k: int = None
):
    query = questions
    query_rel_texts = relevant_texts

    article_texts: str = get_article_text_retrieved_or_labelled(
        cutoff_k,
        query,
        query_rel_texts,
        retrieve_docs,
        retriever,
        reranker,
        reranker_cutoff_k
    )

    gen_evidence_lst: list[str] = []
    for ans in answers:
        llm_input: list[dict[str, str]] = prepare_llm_input(
            prompt_in, [query, ans, article_texts]
        )
        pipe_out, g_succ = generate_text(llm_input, pipeline)
        if g_succ:
            llm_gen_dict = pipe_out[0]["generated_text"][1]
            gen_text = llm_gen_dict["content"]
            llm_answer = clean_llm_output_content(answer_start_sentence, gen_text)
            gen_evidence_lst.append(llm_answer)

    return {"evidence": gen_evidence_lst}


def retrieve_art_ids_for_question(
        questions: list[str],
        retriever: LegalFAISSRetriever,
        cutoff_k: int,
        reranker: Qwen3Reranker = None,
        reranker_cutoff_k: int = None
) -> dict[str, list[list[int]]]:
    retr_art_ids_batch: list[list[int]] = []

    for question in questions:
        retrieved_data = retriever.retrieve(question, k=cutoff_k)
        retr_art_ids: list[int] = [
            int(ret_data["metadata"]["doc_id"]) for ret_data in retrieved_data
        ]

        if reranker is not None:
            retrieved_texts = [r["text"] for r in retrieved_data]
            text_to_id = {
                r["text"]: int(r["metadata"]["doc_id"]) for r in retrieved_data
            }
            reranked_texts = reranker.rerank(
                question, retrieved_texts, return_scores=False
            )
            reranked_texts = reranked_texts[:reranker_cutoff_k]
            retr_art_ids = [text_to_id[text] for text in reranked_texts]

        retr_art_ids_batch.append(retr_art_ids)

    return {"retrieved_art_ids": retr_art_ids_batch}


if __name__ == "__main__":
    args = parse_args()

    corpus_ds: datasets.Dataset = datasets.load_dataset(
        "jurifindit/JuriFindIT", "corpus", split="corpus"
    )
    question_df = pd.read_pickle("datasets/justifitqa.pkl")
    questions_ds = datasets.Dataset.from_pandas(question_df)

    conversation_ds = datasets.load_from_disk(args.conversation_ds_path)
    conversation_ds = conversation_ds.map(
        lambda lst_dict: {
            "gen_answers": [d["content"] for d in eval(lst_dict) if d["role"] == "assistant"]
        },
        input_columns=["conversation"]
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.llm_model_repo,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        device_map="auto",
        dtype="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm_model_repo,
        cache_dir=args.cache_dir,
        padding_side="left",
        model_max_length=model.config.max_position_embeddings - 1
    )

    embedder_model_kwargs = {"device": "cuda", "trust_remote_code": True}
    query_encode_kwargs = {"prompt": "Query: ", "batch_size": args.retriever_bsz}
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedder_model_name,
        cache_folder=args.cache_dir,
        show_progress=False,
        model_kwargs=embedder_model_kwargs,
        query_encode_kwargs=query_encode_kwargs
    )

    splitter = None
    faiss_retriever = LegalFAISSRetriever(
        corpus=corpus_ds,
        embedding=embedding_model,
        splitter=splitter,
        config=BuildConfig(text_field="content", id_field="id"),
    )
    retriever = faiss_retriever.build()

    answer_start = "</think>"
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template,
        enable_thinking=args.use_thinking
    )

    pipeline = transformers.pipelines.TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=8192,
        max_length=None
    )

    reranker = Qwen3Reranker(
        model_name=args.reranker_model_repo,
        cache_dir=args.cache_dir,
        batch_size=args.reranker_bsz
    )

    rnd_fgrprnt = random.randint(0, int(2 ** 31))

    conversation_ds = conversation_ds.map(
        generate_evidence,
        batched=False,
        fn_kwargs={
            "pipeline": pipeline,
            "retriever": retriever,
            "cutoff_k": args.cutoff_k,
            "prompt_in": gen_evidence_prompt,
            "answer_start_sentence": answer_start,
            "retrieve_docs": True,
            "reranker": reranker,
            "reranker_cutoff_k": args.reranker_cutoff_k,
        },
        input_columns=["question", "relevant_doc_txt", "gen_answers"],
        new_fingerprint=f"{rnd_fgrprnt}"
    )

    conversation_ds = conversation_ds.map(
        retrieve_art_ids_for_question,
        batched=True,
        batch_size=1000,
        fn_kwargs={
            "retriever": retriever,
            "cutoff_k": args.cutoff_k,
            "reranker": reranker,
            "reranker_cutoff_k": args.reranker_cutoff_k,
        },
        input_columns=["question"],
        new_fingerprint=f"{rnd_fgrprnt}"
    )

    conversation_ds.save_to_disk(args.output_path)

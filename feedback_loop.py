import argparse
import itertools
import json
import random
import re
import time
from functools import partial

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import TextGenerationPipeline

from Qwen3Reranker import Qwen3Reranker
from build_vector_db import LegalFAISSRetriever, BuildConfig
from ext_dataset_utils import get_cleaned_question_dataset
import datasets
import pandas as pd
import torch
import transformers

italian_compliant_prompt = """Sei un modello esperto di diritto italiano. Il tuo compito è rispondere (in italiano) a una Domanda giuridica (Q) basandoti SOLO sugli Articoli Rilevanti (D) forniti nel contesto.
La tua risposta deve essere giuridicamente precisa, conforme a D, e priva di qualsiasi invenzione o distorsione.

Segui questi principi per garantire la CONFORMITÀ durante la generazione:

C1) Copertura e ancoraggio ai testi:
   - Includi ogni elemento materiale di D che risponde direttamente a Q.
   - Deriva la risposta esclusivamente da quanto è esplicitamente contenuto o chiaramente implicato in D.
   - Se gli articoli non forniscono informazioni sufficienti, dichiaralo chiaramente (es.: "Gli articoli forniti non specificano…").

C2) Nessuna contraddizione:
   - Non contraddire mai il significato, le condizioni o le conseguenze stabilite in D.
   - Non mescolare disposizioni non correlate né fondere clausole applicabili a contesti o giurisdizioni diverse.

C3) Conservazione della modalità e della precisione:
   - Mantieni la stessa **forza deontica** di D: se D impone "deve"/"è tenuto a", non sostituirlo con "può"/"dovrebbe".
   - Mantieni esatti i valori quantitativi e temporali (es.: "30 giorni" ≠ "45 giorni"; "prima dell’approvazione" ≠ "dopo l’approvazione").

C4) Conservazione dell’ambito di applicazione:
   - Non alterare o generalizzare **chi** è vincolato (parti, autorità, soggetti), **dove** (giurisdizione) o **quando** (validità temporale).
   - Riporta condizioni, eccezioni e definizioni esattamente come fornite.

C5) Sintesi e integrazione fedeli:
   - Puoi riassumere o riformulare le norme di D per chiarezza, senza modificarne il significato.
   - Puoi aggiungere frasi di collegamento, definizioni o esempi solo se direttamente inferibili da D.
   - Non includere fatti esterni, precedenti o interpretazioni non presenti o non implicate in D.

C6) Trasparenza:
   - Cita o parafrasa in modo fedele le espressioni giuridiche chiave.
   - Se pertinente, indica l’id dell’articolo (es.: "Ai sensi dell’articolo 5…").

Se vi è ambiguità o mancano informazioni, non speculare.
Dichiara invece che la risposta non può essere determinata integralmente sulla base degli articoli forniti.

INPUT
Q (domanda):
<<INIZIO DOMANDA>>
{0}
<<FINE DOMANDA>>

D (articoli rilevanti come oggetti JSON con "id" e "content"):
<<INIZIO TESTI DEGLI ARTICOLI>>
{1}
<<FINE TESTI DEGLI ARTICOLI>>

OUTPUT
IMPORTANTE: RISPONDI IN ITALIANO!

Scrivi una risposta chiara e giuridicamente precisa, in linguaggio naturale, che:
1) Risponda direttamente a Q.
2) Sia rigorosamente ancorata ai documenti D.
3) Preservi significato giuridico, modalità, ambito e dettagli quantitativi.
4) CITI esplicitamente i fatti rilevanti e gli articoli di D (es.: "Art. X" o "id: …").

"""

italian_compliant_text_lasso_v2 = '''Sei un verificatore giuridico esperto.  
Il tuo compito è analizzare se una Risposta (R) rispetti fedelmente i contenuti e il significato giuridico degli Articoli Rilevanti (D) forniti come contesto.

Devi individuare con precisione **quali parti della risposta non sono supportate, contraddicono o distorcono** le disposizioni contenute in D.  
Il tuo obiettivo è fornire un **feedback chiaro e mirato**, che permetta di correggere la risposta per renderla pienamente conforme ai documenti legali.

---

### CRITERI DI VERIFICA

C1) **Copertura:** ogni affermazione rilevante nella risposta deve essere supportata o implicata dai documenti D.  
C2) **Assenza di contraddizioni:** la risposta non deve contraddire alcuna disposizione, condizione o regola di D.  
C3) **Modalità e forza normativa:** la risposta deve mantenere la stessa forza deontica e semantica di D (es.: “deve” ≠ “può”; “vietato” ≠ “consentito”), conservando i valori quantitativi e temporali originali.  
C4) **Ambito di applicazione:** non devono esserci variazioni rispetto a soggetti, giurisdizioni, periodi di validità, eccezioni, definizioni o condizioni espresse in D.  
C5) **Fedele integrazione:** eventuali estensioni o parafrasi devono essere coerenti con D e non introdurre informazioni nuove o arbitrarie.

---

### ISTRUZIONI OPERATIVE

- Esamina attentamente i documenti D e la risposta R.  
- Determina se la risposta è conforme, parzialmente conforme o non conforme, in base ai criteri sopra elencati:  
  - **CONFORME:** tutti i criteri C1–C5 sono rispettati.  
  - **PARZIALMENTE CONFORME:** C1–C4 sono rispettati, ma C5 è violato (la risposta è nel complesso coerente ma include dettagli aggiuntivi non esplicitamente presenti in D).  
  - **NON CONFORME:** uno o più dei criteri C1–C4 non sono rispettati (contraddizioni, errori, variazioni di modalità o ambito).  

- Se la risposta non è pienamente conforme, fornisci un feedback testuale che indichi **esattamente cosa deve essere modificato o eliminato** per raggiungere la conformità.  

Il feedback deve concentrarsi su:
- affermazioni presenti in R **non giustificate o non supportate** da D;  
- contraddizioni o interpretazioni errate;  
- omissioni di elementi giuridicamente rilevanti presenti in D;  
- variazioni di forza normativa, ambito o condizione.

Evita commenti generici: ogni osservazione deve essere **basata su prove testuali** tratte dai documenti D.  
Se necessario, cita brevemente il punto del documento D a cui il feedback si riferisce.

---

### INPUT

D (articoli rilevanti come oggetti JSON con "id" e "content"):  
<<INIZIO ARTICOLI RILEVANTI>>  
{1}  
<<FINE ARTICOLI RILEVANTI>>

R (risposta da valutare):  
<<INIZIO RISPOSTA>>  
{0}  
<<FINE RISPOSTA>>

---

### OUTPUT (restituisci SOLO questo oggetto JSON)

{{
  "evaluation": "CONFORME" | "PARZIALMENTE CONFORME" | "NON CONFORME",
  "feedback": "<descrizione dettagliata dei problemi riscontrati e istruzioni su come correggerli per rendere la risposta conforme>"
}}

---

### POLITICA DECISIONALE (solo per ragionamento interno, non includere nell’output)

- **CONFORME:** tutti i criteri C1–C5 sono soddisfatti → nessuna modifica necessaria.  
- **PARZIALMENTE CONFORME:** C1–C4 soddisfatti, ma C5 violato → suggerisci miglioramenti per rendere la risposta più aderente e meno estesa rispetto a D.  
- **NON CONFORME:** una o più violazioni dei criteri C1–C4 → fornisci feedback chiari e puntuali su cosa va corretto o rimosso per evitare errori sostanziali rispetto a D.'''

feedback_prompt_v2 = '''Hai già partecipato a questa conversazione giuridica, in cui ti è stata posta una domanda e hai fornito una risposta basandoti sugli articoli legali presenti nel contesto.  
Ora hai ricevuto un feedback da un verificatore che segnala errori, imprecisioni o punti da migliorare nella tua risposta.

Il tuo compito è **migliorare la tua risposta precedente**, tenendo conto del feedback ricevuto e del contesto della conversazione.

---

### ISTRUZIONI

1. Analizza attentamente il feedback ricevuto.  
   - Identifica cosa deve essere corretto, chiarito o approfondito.  
   - Presta attenzione a eventuali punti in cui la tua risposta non è supportata dagli articoli o risulta parzialmente imprecisa.

2. Riscrivi o integra la tua risposta **direttamente nel flusso della conversazione**, producendo una nuova versione che:
   - **risponda in modo diretto e completo alla domanda originale**;  
   - **sia pienamente conforme** ai documenti legali nel contesto;  
   - **corregga tutte le incongruenze o omissioni** segnalate nel feedback;  
   - mantenga **tono, chiarezza e coerenza** con la conversazione in corso.

3. Non introdurre informazioni nuove o speculative che non siano presenti o chiaramente implicate negli articoli forniti.  
4. Se il feedback chiede di chiarire o espandere, fallo solo restando entro i limiti del testo legale di riferimento.  
5. Assicurati che la nuova risposta rimanga coerente con il dialogo precedente e mantenga un linguaggio giuridico formale ma naturale.

---

### INPUT

**Feedback del verificatore:**  
<<INIZIO FEEDBACK>>  
{0}  
<<FINE FEEDBACK>>

**Domanda iniziale Q:**  
<<INIZIO DOMANDA>>
{1}
<<FINE DOMANDA>>
---

### OUTPUT

Scrivi la **nuova versione della tua risposta** che:
- corregga i punti indicati nel feedback;  
- resti coerente con la domanda iniziale e il contesto della chat;  
- sia pienamente conforme agli articoli legali presenti nella conversazione.  

IMPORTANTE:  
Riformula la risposta come se stessi **continuando la conversazione**, non come se stessi scrivendo da zero o ripetendo il prompt.  
Il tuo obiettivo è dare una **risposta migliore, più precisa e legalmente corretta**, restando nel tono del dialogo in corso.
'''

prompt_coherence_labelled_gen_question_docs = '''Sei un verificatore giuridico meticoloso. Il tuo compito è determinare se la Risposta G sia **CONFORME** alla Risposta L, consultando il contenuto dei documenti D.  
Devi valutare se G rispecchi fedelmente, riassuma o ampli il contenuto di L **senza modificarne il significato giuridico, la forza normativa o l’ambito di applicazione**.

Devi decidere se G è:
- **CONFORME:** quando tutte le affermazioni in G sono coerenti con L, senza contraddizioni, né alterazioni di ambito o modalità.  
  G può includere estensioni, riformulazioni o chiarimenti rispetto a L, purché queste aggiunte siano **coerenti con il contenuto e il significato giuridico di L**, senza introdurre nuovi obblighi, eccezioni o interpretazioni incompatibili.  
- **NON CONFORME:** quando G contraddice, modifica o distorce in modo sostanziale L, altera la forza normativa, l’ambito di applicazione o aggiunge elementi inconciliabili con quanto espresso in L.

Valuta la CONFORMITÀ utilizzando i seguenti criteri:

C1) Nessuna contraddizione: G non deve contraddire alcuna disposizione, condizione o conclusione presente in L.  
C2) Nessuna variazione di modalità: G deve mantenere la **forza deontica e la polarità** di L — es.: "deve"/"è tenuto a" ≠ "può"/"dovrebbe"; "vietato" ≠ "consentito"; e tutti gli **elementi quantitativi** (numeri, termini temporali, soglie) devono rimanere identici.  
C3) Nessuna variazione di ambito: G non deve modificare **soggetti, giurisdizioni, periodi di validità, eccezioni, definizioni o condizioni** espressi in L.  

Regole generali:
- Considera il contenuto dei testi G e L, tenendo in considerazione i documenti D forniti.  
- Se un’affermazione di G non è supportata da L ma è compatibile con esso e trova riscontro nei documenti, può comunque essere considerata conforme.  
- G può **estendere** il contenuto di L con fatti forniti in D, ma non può **contraddirlo**, **invertirne la forza normativa** o **cambiarne le condizioni applicative**.  
- Concentrati sul significato giuridico sostanziale, non sulle differenze di stile o forma linguistica.  
- **Ignora errori formali o di citazione**, soprattutto riferiti al numero degli articoli, se non influiscono sul contenuto sostanziale.

---

### INPUT

L (testo di riferimento):  
<<INIZIO TESTO L>>  
{0}  
<<FINE TESTO L>>

G (testo da verificare):  
<<INIZIO TESTO G>>  
{1}  
<<FINE TESTO G>>

D (documenti rilevanti):
<<INIZIO TESTO D>>
{2}
<<FINE TESTO D>>
---

### OUTPUT (restituisci SOLO questo oggetto JSON):

{{
  "evaluation": "CONFORME" | "NON CONFORME",
  "reason": "<giustificazione concisa che faccia riferimento ai criteri C1–C3>",
  "evidence": {{
    "contradictions": [
      {{"l_span": "<testo in L>", "g_span": "<testo in G>", "note": "<spiegazione del conflitto>"}}
    ],
    "incoherences": [
      {{"g_claim": "<testo in G>", "note": "<elemento incoerente o non compatibile con L>"}}
    ],
    "supporting_facts": [
      {{"l_span": "<testo in L>", "g_span": "<testo in G>", "note": "<coerenza o corrispondenza sostanziale>"}}
    ]
  }}
}}

'''


def extract_evaluation(text: str) -> tuple[str, float]:
    """
    Extracts the value associated with the "evaluation" key from a possibly malformed JSON-like string.
    Returns the value as a string, or None if not found.
    """
    # Regex explanation:
    # - (?i) makes it case-insensitive (matches "Evaluation", "evaluation", etc.)
    # - looks for "evaluation" followed by ":" and optional spaces
    # - captures the value between quotes (either single or double)
    result_mapping = {"compliant": 1.0, "semi-compliant": 0.5, "error": 0.0, "non-compliant": 0.0,
                      "conforme": 1.0, "parzialmente conforme": 0.5, "non conforme": 0.0}
    pattern = r'(?i)"evaluation"\s*:\s*["\']([^"\']+)["\']'

    match = re.search(pattern, text)
    result = "ERROR"
    if match:
        result = match.group(1).strip().lower()
    num_out = result_mapping.get(result, 0)
    if result == "ERROR" or result not in result_mapping:
        print("ERROR IN EXTRACTING EVALUATION!")
    return (result, float(num_out))


def clean_gen_texts(gen_txt_lst: list[str]) -> list[str]:
    gen_text = [re.sub(r'[\u202f\u00a0\u2007]', ' ', txt.strip()) for txt in gen_txt_lst]
    # gen_text = [txt.replace("`", "").replace("markdown", "").strip() for txt in gen_text]
    return gen_text


def get_only_assistant_text_from_conversation(conversation: str) -> list[str]:
    parsed_conv: list[dict[str, str]] = json.loads(conversation)
    assistant_texts = []
    for role_cont_dict in parsed_conv:
        if role_cont_dict['role'] == 'assistant':
            assistant_texts.append(role_cont_dict['content'])
    assistant_texts = assistant_texts if len(assistant_texts) > 0 else ["Nessuna risposta."]
    return assistant_texts


def get_article_text_retrieved_or_labelled(cutoff_k, query, query_rel_texts, retrieve_docs, retriever, reranker,
                                           reranker_cutoff_k):
    if retrieve_docs:
        article_texts: str = retrieve_and_format_articles_in_json(query, retriever, cutoff_k, reranker,
                                                                  reranker_cutoff_k)
    else:
        # query_rel_texts = relevant_texts[i]
        cut_k = cutoff_k if len(query_rel_texts) > cutoff_k else len(query_rel_texts)
        query_rel_texts = query_rel_texts[:cut_k]
        article_texts: str = format_articles_in_json(query_rel_texts)
    return article_texts


def evaluate_gen_labelled_answers_coherence(questions: list[str], relevant_texts: list[list[str]], answers: list[str],
                                            conversation: list[str], prompt_verifier: str,
                                            verifier_pipeline: TextGenerationPipeline,
                                            cutoff_k: int, answer_start_sentence: str):
    evaluations = []
    for i in range(len(questions)):
        # query = questions[i]
        labelled_answer = answers[i]
        query_rel_texts = relevant_texts[i]
        cut_k = cutoff_k if len(query_rel_texts) > cutoff_k else len(query_rel_texts)
        query_rel_texts = query_rel_texts[:cut_k]
        article_texts: str = format_articles_in_json(query_rel_texts)
        assistant_texts = get_only_assistant_text_from_conversation(conversation[i])
        verifier_input_batch = []
        for response in assistant_texts:
            verifier_input: list[dict[str, str]] = prepare_llm_input(prompt_verifier,
                                                                     [labelled_answer, response, article_texts])
            verifier_input_batch.append(verifier_input)

        verifier_out, v_succ = generate_text(verifier_input_batch, verifier_pipeline)
        batch_evals = []
        for a_i in range(len(verifier_out)):
            verifier_out_i = verifier_out[a_i]
            numeric_decision = 0.0
            if v_succ:
                verifier_answer = verifier_out_i[0]['generated_text'][1]['content']
                verifier_answer = clean_llm_output_content(answer_start_sentence, verifier_answer)
                _, numeric_decision = extract_evaluation(verifier_answer)
            batch_evals.append(numeric_decision)
        evaluations.append(batch_evals)
        print(batch_evals)
    return {"evaluations_lbl_gen_answers": evaluations}


def generate_synth_answers_history(questions: list[str], relevant_texts: list[list[str]], answers: list[str],
                                   pipeline: TextGenerationPipeline, retriever: LegalFAISSRetriever,
                                   prompt_in: str, prompt_verifier: str, feedback_prompt: str,
                                   verifier_pipeline: TextGenerationPipeline,
                                   cutoff_k: int, answer_start_sentence: str, max_iters: int,
                                   retrieve_docs: bool = True, reranker: Qwen3Reranker = None,
                                   reranker_cutoff_k: int = None):
    query = questions[0]
    query_rel_texts = relevant_texts[0]
    intermediate_results: list[float] = []

    article_texts: str = get_article_text_retrieved_or_labelled(cutoff_k, query, query_rel_texts, retrieve_docs,
                                                                retriever, reranker, reranker_cutoff_k)

    llm_input: list[dict[str, str]] = prepare_llm_input(prompt_in, [query, article_texts])
    iter_num = 0
    gen_answ_is_ok = False
    while iter_num < max_iters and not gen_answ_is_ok:
        pipe_out, g_succ = generate_text(llm_input, pipeline)
        numeric_decision = 0
        if g_succ:
            llm_gen_dict = pipe_out[0]['generated_text'][iter_num * 2 + 1]
            gen_text = llm_gen_dict['content']
            llm_answer = clean_llm_output_content(answer_start_sentence, gen_text)
            llm_gen_dict['content'] = llm_answer
            llm_input.append(llm_gen_dict)
            verifier_input = prepare_llm_input(prompt_verifier, [llm_answer, article_texts])
            verifier_out, v_succ = generate_text(verifier_input, verifier_pipeline)
            if v_succ:
                verifier_answer = verifier_out[0]['generated_text'][1]['content']
                verifier_answer = clean_llm_output_content(answer_start_sentence, verifier_answer)
                decision, numeric_decision = extract_evaluation(verifier_answer)

                if numeric_decision < 1.0:
                    llm_input.append({"role": "user", "content": feedback_prompt.format(verifier_answer, query)})
                else:
                    gen_answ_is_ok = True

                intermediate_results.append(numeric_decision)
            else:
                dummy_feedback = "Risposta errata. Genera una risposta piu precisa."
                llm_input.append({"role": "user",
                                  "content": feedback_prompt.format(dummy_feedback, query)})
        else:
            break

        result_str = f"\nIter: {iter_num + 1} => {numeric_decision}"
        result_str = result_str.lstrip() if iter_num > 0 else result_str
        print(result_str)
        iter_num += 1
    print()
    return {"conversation": [json.dumps(llm_input)], "eval_history": [intermediate_results]}


def clean_llm_output_content(answer_start_sentence, gen_text) -> str:
    gen_text = gen_text.split(answer_start_sentence)[1] if answer_start_sentence in gen_text else gen_text
    llm_answer = clean_gen_texts([gen_text])[0]
    return llm_answer


def generate_text(batched_instr_lst, pipeline):
    batch_size = len(batched_instr_lst)
    pipe_out = None
    loop_for_memory = True
    success_trace = False
    while loop_for_memory:
        try:
            pipe_out = pipeline(batched_instr_lst, return_full_text=True,
                                truncation=True, batch_size=batch_size, num_workers=1)
            success_trace = True
            loop_for_memory = False
        except torch.OutOfMemoryError:
            batch_size = batch_size // 2
            print(f"Retrying with batch size of {batch_size}.")
            if batch_size == 0:
                print("Batch size is too small.")
                pipe_out = [[{"generated_text": ["Too large contex."] * len(batched_instr_lst)}]]
                loop_for_memory = False
    return (pipe_out, success_trace)


def retrieve_and_format_articles_in_json(query: str, retriever: LegalFAISSRetriever, cutoff_k: int,
                                         reranker: Qwen3Reranker, reranker_cutoff_k: int) -> str:
    query_rel_txt: list[str] = retriever.retrieve_text_only(query, cutoff_k)  # relevant_texts[i]
    if reranker:
        query_rel_txt = reranker.rerank(query, query_rel_txt, return_scores=False)
        query_rel_txt = query_rel_txt[:reranker_cutoff_k]

    articles_text = format_articles_in_json(query_rel_txt)
    return articles_text


def format_articles_in_json(query_rel_txt):
    articles_text = ",\n".join(
        "{{'doc_id': {idx}, 'content': {txt} }}".format(idx=i + 1, txt=txt) for i, txt in enumerate(query_rel_txt))
    articles_text = f"```json\n[\n{articles_text}\n]\n```"
    return articles_text


def prepare_llm_input(prompt_in: str, fill_args: list[str]) -> list[dict[str, str]]:
    filled_prompt = prompt_in.format(*fill_args)
    return [{"role": "user", "content": filled_prompt}]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feedback-loop generation and evaluation pipeline."
    )

    parser.add_argument(
        "--llm_model_repo",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF repository of the main LLM."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".",
        help="Cache directory for HF models/tokenizers/embeddings."
    )
    parser.add_argument(
        "--llm_verifier_repo",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF repository of the verifier LLM."
    )
    parser.add_argument(
        "--embedder_model_name",
        type=str,
        default="retriever",
        help="Embedding model name/path."
    )
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        help="Enable thinking mode for the main LLM chat template."
    )
    parser.add_argument(
        "--use_thinking_verif",
        action="store_true",
        help="Enable thinking mode for the verifier LLM chat template."
    )
    parser.add_argument(
        "--cutoff_k",
        type=int,
        default=20,
        help="Top-k documents retrieved before answer generation."
    )
    parser.add_argument(
        "--reranker_cutoff_k",
        type=int,
        default=4,
        help="Top-k documents kept after reranking."
    )
    parser.add_argument(
        "--feedback_loop_max_iters",
        type=int,
        default=3,
        help="Maximum number of feedback-loop iterations."
    )
    parser.add_argument(
        "--reranker_model_repo",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="HF repository of the reranker model."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    corpus_ds: datasets.Dataset = datasets.load_dataset(
        "jurifindit/JuriFindIT", "corpus", split="corpus"
    )
    question_df = pd.read_parquet("datasets/justifitqa.pkl")
    questions_ds = datasets.Dataset.from_pandas(question_df)

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
        model_max_length=int(2 ** 15)
    )

    tokenizer_verif = transformers.AutoTokenizer.from_pretrained(
        args.llm_verifier_repo,
        cache_dir=args.cache_dir,
        padding_side="left"
    )

    embedder_model_kwargs = {"device": "cuda", "trust_remote_code": True}
    query_encode_kwargs = {"prompt": "Query: ", "batch_size": 16}
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

    model_verif = transformers.AutoModelForCausalLM.from_pretrained(
        args.llm_verifier_repo,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        device_map="auto",
        dtype="auto"
    )
    tokenizer_verif.apply_chat_template = partial(
        tokenizer_verif.apply_chat_template,
        enable_thinking=args.use_thinking_verif
    )

    pipeline_verif = transformers.pipelines.TextGenerationPipeline(
        model=model_verif,
        tokenizer=tokenizer_verif,
        max_new_tokens=8192,
        max_length=None
    )

    question_df["relevant_text_sections"] = question_df["relevant_text_sections"].map(
        lambda rel_dict: list(itertools.chain.from_iterable(rel_dict.values()))
    )

    fgp = random.randint(0, int(2 ** 31))
    reranker = Qwen3Reranker(model_name=args.reranker_model_repo)
    formed_prompt = italian_compliant_prompt

    conversation_ds = questions_ds.map(
        generate_synth_answers_history,
        batched=True,
        batch_size=1,
        fn_kwargs={
            "pipeline": pipeline,
            "prompt_in": formed_prompt,
            "cutoff_k": args.cutoff_k,
            "answer_start_sentence": answer_start,
            "retriever": retriever,
            "retrieve_docs": True,
            "prompt_verifier": italian_compliant_text_lasso_v2,
            "feedback_prompt": feedback_prompt_v2,
            "verifier_pipeline": pipeline,
            "reranker": reranker,
            "reranker_cutoff_k": args.reranker_cutoff_k,
            "max_iters": args.feedback_loop_max_iters,
        },
        input_columns=["question", "relevant_text_sections", "answer"],
        new_fingerprint=f"{fgp}_feedback_loop"
    )
    conversation_ds.save_to_disk("datasets/feedback_loop_result")

    conversation_ds = conversation_ds.map(
        evaluate_gen_labelled_answers_coherence,
        batched=True,
        batch_size=1,
        fn_kwargs={
            "cutoff_k": args.cutoff_k,
            "answer_start_sentence": answer_start,
            "prompt_verifier": prompt_coherence_labelled_gen_question_docs,
            "verifier_pipeline": pipeline,
        },
        input_columns=["question", "relevant_doc_txt", "answer", "conversation"],
        new_fingerprint=f"{fgp}_evaluate_answers"
    )
    conversation_ds.save_to_disk("datasets/feedback_loop_result_w_answer_eval")

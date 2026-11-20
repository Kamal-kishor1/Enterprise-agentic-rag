import asyncio
import time
import json
from app.agents.graph import rag_agent
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.services.cache_service import get_cached_query, set_cached_query

test_cases = [
    {
        "id": "TC-01",
        "category": "Technical (Autoscaling)",
        "query": "How does Horizontal Pod Autoscaler work in Kubernetes?",
        "target_doc": "pods_autoscale.html"
    },
    {
        "id": "TC-02",
        "category": "Technical (CronJobs)",
        "query": "How do CronJobs handle concurrency policies in Kubernetes?",
        "target_doc": "cronjobs.docx"
    },
    {
        "id": "TC-03",
        "category": "Technical (Work Queue)",
        "query": "Explain coarse parallel processing with a work queue in Kubernetes.",
        "target_doc": "parallel_work_queue.txt"
    },
    {
        "id": "TC-04",
        "category": "Technical (Job Failure)",
        "query": "What happens when a Kubernetes Job fails or exceeds backoffLimit?",
        "target_doc": "monitor_job.docx"
    },
    {
        "id": "TC-05",
        "category": "Technical (Architecture)",
        "query": "What are the core components of the Kubernetes cluster architecture?",
        "target_doc": "architecture.pptx"
    },
    {
        "id": "TC-06",
        "category": "Technical (HPA Metrics)",
        "query": "What metrics can the Horizontal Pod Autoscaler use for scaling pods?",
        "target_doc": "pods_autoscale.html"
    },
    {
        "id": "TC-07",
        "category": "Technical (Job Completion)",
        "query": "How do indexed completion modes work in Kubernetes Jobs?",
        "target_doc": "job_management.html"
    },
    {
        "id": "TC-08",
        "category": "Technical (Worker Termination)",
        "query": "How does a worker pod know when a parallel work queue job is finished?",
        "target_doc": "parallel_work_queue.txt"
    },
    {
        "id": "TC-09",
        "category": "Conversational (Greeting)",
        "query": "Hello there! What kind of assistant are you and what topics do you know?",
        "target_doc": None
    },
    {
        "id": "TC-10",
        "category": "Conversational (General Query)",
        "query": "Can you give me general advice on organizing cloud native infrastructure?",
        "target_doc": None
    }
]

async def benchmark_run():
    print("=" * 80)
    print("RUNNING POST-IMPROVEMENT BENCHMARK ON 10 TEST CASES")
    print("=" * 80)
    
    results = []
    
    for tc in test_cases:
        qid = tc["id"]
        q = tc["query"]
        cat = tc["category"]
        target = tc["target_doc"]
        print(f"\n[{qid}] {cat}: '{q[:50]}...'")
        
        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "documents": [],
            "plan": ["Start"],
            "status": "Init",
        }
        config = {"configurable": {"thread_id": f"bench-post-{qid.lower()}"}}
        
        # Measure TTFT and Total E2E streaming latency
        t0 = time.perf_counter()
        first_token_time = None
        tokens = []
        
        try:
            async for event in rag_agent.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "responder":
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - t0
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            tokens.append(chunk)
            
            t_total = time.perf_counter() - t0
            full_answer = "".join(tokens)
            ttft = first_token_time if first_token_time else t_total
            
            # Check retrieval hit if technical
            hit1, hit3, hit5 = False, False, False
            if target:
                docs = search_enterprise_knowledge(q, limit=5)
                sources = [d.get("source", "") for d in docs]
                hit1 = (target in sources[0]) if sources else False
                hit3 = any(target in s for s in sources[:3])
                hit5 = any(target in s for s in sources[:5])
            
            out_chars = len(full_answer)
            out_tokens = out_chars // 4
            
            entry = {
                "id": qid,
                "category": cat,
                "query": q,
                "target_doc": target,
                "ttft": round(ttft, 2),
                "e2e_latency": round(t_total, 2),
                "output_chars": out_chars,
                "output_tokens": out_tokens,
                "hit1": hit1,
                "hit3": hit3,
                "hit5": hit5,
                "status": "success",
                "answer_snippet": full_answer[:120].replace("\n", " ")
            }
            print(f"   -> TTFT: {ttft:.2f}s | E2E: {t_total:.2f}s | Tokens: ~{out_tokens} | Hit@1: {hit1} | Hit@5: {hit5}")
            results.append(entry)
            
        except Exception as e:
            print(f"   -> ERROR: {e}")
            results.append({
                "id": qid,
                "category": cat,
                "query": q,
                "target_doc": target,
                "ttft": 0.0,
                "e2e_latency": 0.0,
                "output_chars": 0,
                "output_tokens": 0,
                "hit1": False,
                "hit3": False,
                "hit5": False,
                "status": f"error: {e}",
                "answer_snippet": ""
            })
            
        # Pause 3.0s between queries to avoid Groq TPM limit bursts
        await asyncio.sleep(3.0)
        
    with open("docs/post_improvement_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark saved to docs/post_improvement_benchmark.json")

if __name__ == "__main__":
    asyncio.run(benchmark_run())

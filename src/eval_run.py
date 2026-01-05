"""Evaluation script for RAG Assistant.

Runs sample questions and logs retrieval results and answers.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from src.config import get_config, get_project_root
from src.exceptions import RAGException
from src.logging_config import configure_logging, get_logger
from src.rag_chat import RAGAssistant, RAGResponse

logger = get_logger(__name__)


# Sample evaluation questions
SAMPLE_QUESTIONS = [
    "What are the main topics covered in the documents?",
    "Can you summarize the key points from the documents?",
    "What important concepts or terms are defined in the documents?",
    "Are there any procedures or steps described in the documents?",
    "What conclusions or recommendations are mentioned in the documents?",
]


def run_evaluation(
    questions: List[str] | None = None,
    output_file: str | None = None,
) -> List[RAGResponse]:
    """Run evaluation with sample questions.
    
    Args:
        questions: List of questions to ask. Uses defaults if None.
        output_file: Path to save results. Uses default if None.
        
    Returns:
        List of RAGResponse objects.
    """
    if questions is None:
        questions = SAMPLE_QUESTIONS
    
    config = get_config()
    project_root = get_project_root()
    logs_dir = project_root / config.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    configure_logging(
        log_dir=logs_dir,
        log_level=config.logging.level,
    )
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = logs_dir / f"eval_results_{timestamp}.log"
    else:
        output_file = Path(output_file)
    
    print("=" * 70)
    print("RAG Assistant - Evaluation Run")
    print("=" * 70)
    print(f"Questions: {len(questions)}")
    print(f"Model: {config.llm.model_name}")
    print(f"Output file: {output_file}")
    print("=" * 70 + "\n")
    
    # Initialize assistant
    try:
        assistant = RAGAssistant()
    except RAGException as e:
        logger.error(f"Failed to initialize assistant: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    
    results: List[RAGResponse] = []
    eval_log_lines: List[str] = []
    
    eval_log_lines.append("=" * 70)
    eval_log_lines.append(f"RAG Assistant Evaluation - {datetime.now().isoformat()}")
    eval_log_lines.append(f"Model: {config.llm.model_name}")
    eval_log_lines.append(f"Embeddings: {config.embeddings.provider}/{config.embeddings.model}")
    eval_log_lines.append(f"Top-K: {config.retrieval.top_k}")
    eval_log_lines.append("=" * 70)
    eval_log_lines.append("")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        print("-" * 50)
        
        eval_log_lines.append(f"\n{'='*70}")
        eval_log_lines.append(f"QUESTION {i}: {question}")
        eval_log_lines.append("=" * 70)
        
        try:
            response = assistant.ask(question)
            results.append(response)
            
            # Log sources
            eval_log_lines.append("\nRETRIEVED SOURCES:")
            for j, source in enumerate(response.sources, 1):
                eval_log_lines.append(
                    f"  [{j}] {source.citation} (score: {source.score:.4f})"
                )
                print(f"  Source {j}: {source.citation} (score: {source.score:.4f})")
            
            # Log answer
            eval_log_lines.append(f"\nANSWER:\n{response.answer}")
            print(f"\nAnswer: {response.answer[:300]}...")
            
            # Log latency
            eval_log_lines.append(f"\nLATENCY: {response.latency_ms:.0f}ms")
            
            # Log source details
            eval_log_lines.append("\nSOURCE DETAILS:")
            for j, source in enumerate(response.sources, 1):
                eval_log_lines.append(f"\n  [{j}] {source.citation}")
                eval_log_lines.append(f"      Content: {source.content[:200]}...")
            
        except RAGException as e:
            error_msg = f"Error processing question: {e}"
            logger.error(error_msg)
            print(f"\n{error_msg}")
            eval_log_lines.append(f"\nERROR: {error_msg}")
    
    # Write log file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_log_lines))
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Questions processed: {len(results)}/{len(questions)}")
    print(f"Results saved to: {output_file}")
    
    # Calculate statistics
    if results:
        # Average latency
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        print(f"Average latency: {avg_latency:.0f}ms")
        
        # Average scores
        all_scores = [s.score for r in results for s in r.sources]
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"Average similarity score: {avg_score:.4f}")
        
        # Sources per answer
        avg_sources = sum(len(r.sources) for r in results) / len(results)
        print(f"Average sources per answer: {avg_sources:.1f}")
    
    print("=" * 70)
    
    return results


def run_custom_eval(questions_file: str) -> List[RAGResponse]:
    """Run evaluation with questions from a file.
    
    Args:
        questions_file: Path to a JSON or text file with questions.
        
    Returns:
        List of RAGResponse objects.
    """
    questions_path = Path(questions_file)
    
    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_file}")
        sys.exit(1)
    
    # Load questions
    if questions_path.suffix == ".json":
        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and "questions" in data:
                questions = data["questions"]
            else:
                print("Error: JSON file must contain a list or {'questions': [...]}")
                sys.exit(1)
    else:
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    
    return run_evaluation(questions)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run evaluation on the RAG Assistant"
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=str,
        help="Path to a file with custom questions (JSON or text)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save evaluation results",
    )
    
    args = parser.parse_args()
    
    if args.questions:
        run_custom_eval(args.questions)
    else:
        run_evaluation(output_file=args.output)

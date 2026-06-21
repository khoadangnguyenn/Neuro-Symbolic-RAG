"""
Embedding logic rules and formulas to GraphDB for seeding the EXACT pipeline.

"""
import argparse
import logging
import hashlib
import sys
import os
import json

# Add parent directory of exact_pipeline to sys.path before importing exact_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exact_pipeline.orchestration.pipeline import ExactPipeline
from exact_pipeline.knowledge.knowledge import FormulaCard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_logic_edges(logic_db):
    logger.info("Building Directed Edges for Logic Graph...")
    graph = logic_db.graph
    nodes = list(graph.nodes(data=True))
    edges_added = 0
    
    for i in range(len(nodes)):
        node_A_id, data_A = nodes[i]
        if "conclusion_predicates" not in data_A:
            continue
            
        try:
            conclusions_A = set(json.loads(data_A["conclusion_predicates"]))
        except:
            conclusions_A = set()
            
        if not conclusions_A:
            continue
            
        for j in range(len(nodes)):
            if i == j:
                continue
                
            node_B_id, data_B = nodes[j]
            if "premise_predicates" not in data_B:
                continue
                
            try:
                premises_B = set(json.loads(data_B["premise_predicates"]))
            except:
                premises_B = set()
                
            # If conclusion of A intersects with premise of B, A feeds into B
            if conclusions_A.intersection(premises_B):
                graph.add_edge(node_A_id, node_B_id, relation="feeds_into")
                edges_added += 1

    logic_db.save_graph()
    logic_db._update_pagerank()
    logger.info(f"Added {edges_added} 'feeds_into' edges to Logic Graph.")

def parse_logic(pipeline: ExactPipeline, limit: int = None):
    logger.info(f"Extracting logic rules using LLM (limit={limit})...")
    logic_db = pipeline.logic.logic_knowledge_index
    llm = pipeline.logic.llm
    
    if not llm or not llm.enabled:
        logger.error("LLM is not enabled. Please set EXACT_LLM_BASE_URL (e.g. http://localhost:8001)")
        return
        
    logic_examples = pipeline.logic_examples
    
    rules_added = 0
    rules_merged = 0
    seen_rules = set(logic_db.graph.nodes())
    
    system_prompt = "You are a logician. Extract the premise predicates and conclusion predicates from the given Logical rule."
    
    for idx, ex in enumerate(logic_examples):
        if limit is not None and (rules_added + rules_merged) >= limit:
            break
            
        for nl, fol in zip(ex.premises_nl, ex.premises_fol):
            if "If " in nl or "ForAll" in fol:
                rule_text = f"Pattern: {nl} -> {fol}"
                rule_hash = hashlib.md5(rule_text.encode()).hexdigest()
                rule_id = f"logic_rule_{rule_hash}"
                
                if rule_id not in seen_rules:
                    # Deduplication lookahead
                    hits = logic_db.search(rule_text, k=1)
                    if hits and hits[0].vector_score > 0.95:
                        matched_id = logic_db.id_fn(hits[0].item)
                        logger.info(f"Deduplicating Logic: Match found with score {hits[0].vector_score:.2f}")
                        
                        user_prompt = f"""
                        Rule: {rule_text}
                        
                        Extract the logical predicates from this rule.
                        You MUST output ONLY a valid JSON object. Do not include markdown formatting or explanations.
                        Use the following keys:
                        - "premise_predicates": list of strings (the conditions)
                        - "conclusion_predicates": list of strings (the results)
                        """
                        try:
                            resp = llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, max_tokens=4096)
                            if resp and matched_id in logic_db.graph.nodes:
                                node_data = logic_db.graph.nodes[matched_id]
                                
                                existing_pre = set(json.loads(node_data.get("premise_predicates", "[]")))
                                existing_pre.update(resp.get("premise_predicates", []))
                                node_data["premise_predicates"] = json.dumps(list(existing_pre))
                                
                                existing_con = set(json.loads(node_data.get("conclusion_predicates", "[]")))
                                existing_con.update(resp.get("conclusion_predicates", []))
                                node_data["conclusion_predicates"] = json.dumps(list(existing_con))
                                
                                rules_merged += 1
                                seen_rules.add(rule_id)
                        except Exception as e:
                            logger.error(f"Failed to process and merge logic rule: {e}")
                        continue

                    # If no duplicate, create new node
                    user_prompt = f"""
                    Rule: {rule_text}
                    
                    Extract the logical predicates from this rule.
                    You MUST output ONLY a valid JSON object. Do not include markdown formatting or explanations.
                    Use the following keys:
                    - "premise_predicates": list of strings (the conditions)
                    - "conclusion_predicates": list of strings (the results)
                    
                    Example for "ForAll(x, completed_A(x) -> qualifies_B(x))":
                    {{
                        "premise_predicates": ["completed_A"],
                        "conclusion_predicates": ["qualifies_B"]
                    }}
                    """
                    try:
                        logger.info(f"Querying LLM for new Logic Rule...")
                        resp = llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, max_tokens=4096)
                        if resp:
                            logic_db.add_rule(rule_id, rule_text, auto_save=False)
                            logic_db.graph.nodes[rule_id]["premise_predicates"] = json.dumps(resp.get("premise_predicates", []))
                            logic_db.graph.nodes[rule_id]["conclusion_predicates"] = json.dumps(resp.get("conclusion_predicates", []))
                            
                            seen_rules.add(rule_id)
                            rules_added += 1
                            if rules_added % 50 == 0:
                                logic_db.save_graph()
                    except Exception as e:
                        logger.error(f"Failed to process logic rule {rule_id}: {e}")
                    
    logger.info(f"Added {rules_added} unique Logic rules, Merged {rules_merged} rules.")
    build_logic_edges(logic_db)

def build_physics_edges(physics_db):
    logger.info("Building Bidirectional Edges for Physics Graph...")
    graph = physics_db.graph
    nodes = list(graph.nodes(data=True))
    edges_added = 0
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_A_id, data_A = nodes[i]
            node_B_id, data_B = nodes[j]
            
            if "variables" not in data_A or "variables" not in data_B:
                continue
                
            try:
                vars_A = set(json.loads(data_A["variables"]))
                vars_B = set(json.loads(data_B["variables"]))
            except:
                continue
                
            if vars_A.intersection(vars_B):
                # Bidirectional wiring
                graph.add_edge(node_A_id, node_B_id, relation="shares_variable")
                graph.add_edge(node_B_id, node_A_id, relation="shares_variable")
                edges_added += 2
                
    physics_db.save_graph()
    physics_db._update_pagerank()
    logger.info(f"Added {edges_added} 'shares_variable' edges to Physics Graph.")

def parse_physics(pipeline: ExactPipeline, limit: int = 50):
    logger.info(f"Extracting physics formulas using LLM (limit={limit})...")
    physics_db = pipeline.physics.physics_knowledge_index
    llm = pipeline.physics.llm
    
    if not llm or not llm.enabled:
        logger.error("LLM is not enabled. Please set EXACT_LLM_BASE_URL (e.g. http://localhost:8001)")
        return
        
    physics_examples = pipeline.physics_examples
    processed = 0
    added = 0
    merged = 0
    seen_formulas = set(physics_db.graph.nodes())
    
    system_prompt = "You are a physics expert. Extract the core physics formula and its variables from the Chain of Thought text."
    
    for idx, ex in enumerate(physics_examples):
        if limit is not None and (added + merged) >= limit:
            break
            
        if not ex.cot:
            continue
            
        # Deduplication lookahead
        # We search with the CoT text to see if an identical formula was already parsed
        hits = physics_db.search(ex.cot, k=1)
        if hits and hits[0].vector_score > 0.95:
            matched_id = physics_db.id_fn(hits[0].item)
            logger.info(f"Deduplicating Physics: Match found with score {hits[0].vector_score:.2f}")
            
            user_prompt = f"""
            Question: {ex.question}
            Chain of Thought: {ex.cot}
            
            Extract the core physics formula used to solve this problem.
            You MUST output ONLY a valid JSON object. Do not include markdown formatting or explanations.
            Use the following keys:
            - "variables": a list of physical variable symbols used (e.g. ["U", "I", "R"])
            """
            try:
                resp = llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, max_tokens=4096)
                if resp and matched_id in physics_db.graph.nodes:
                    node_data = physics_db.graph.nodes[matched_id]
                    existing_vars = set(json.loads(node_data.get("variables", "[]")))
                    existing_vars.update(resp.get("variables", []))
                    node_data["variables"] = json.dumps(list(existing_vars))
                    merged += 1
            except Exception as e:
                logger.error(f"Failed to merge physics rule: {e}")
            
            processed += 1
            continue

        user_prompt = f"""
        Question: {ex.question}
        Chain of Thought: {ex.cot}
        
        Extract the core physics formula used to solve this problem.
        You MUST output ONLY a valid JSON object. Do not include markdown formatting or explanations.
        Use the following keys:
        - "formula_id": a short snake_case name for the formula (e.g. "ohm_law")
        - "family": the general topic (e.g. "electricity", "mechanics")
        - "expression": the mathematical formula (e.g. "U = I * R")
        - "premise": a short sentence describing when to use it
        - "trigger_terms": a list of 3-5 keywords that indicate this formula is needed.
        - "variables": a list of physical variable symbols used (e.g. ["U", "I", "R"])
        """
        
        try:
            logger.info(f"Querying LLM for new Physics question {ex.problem_id}...")
            resp = llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, max_tokens=4096)
            if resp and "formula_id" in resp:
                f_id = resp["formula_id"]
                if f_id not in seen_formulas:
                    card = FormulaCard(
                        formula_id=f_id,
                        family=resp.get("family", "physics"),
                        expression=resp.get("expression", ""),
                        premise=resp.get("premise", ""),
                        trigger_terms=resp.get("trigger_terms", [])
                    )
                    physics_db.add_rule(f_id, card, auto_save=False)
                    # Store variables as node attributes for edge building
                    physics_db.graph.nodes[f_id]["variables"] = json.dumps(resp.get("variables", []))
                    
                    seen_formulas.add(f_id)
                    added += 1
                    if added % 50 == 0:
                        physics_db.save_graph()
                    logger.info(f"Added new formula: {f_id} -> {card.expression}")
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            
    logger.info(f"Added {added} new Physics formulas, Merged {merged} out of {processed} processed problems.")
    build_physics_edges(physics_db)

if __name__ == "__main__":
    # Ensure correct working directory so we can import exact_pipeline
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    parser = argparse.ArgumentParser(description="Auto-Seeder for Knowledge Graph")
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to parse. Leave empty for all.")
    parser.add_argument("--logic-only", action="store_true", help="Only run logic seeder")
    parser.add_argument("--physics-only", action="store_true", help="Only run physics seeder")
    args = parser.parse_args()
    
    pipeline = ExactPipeline()
    
    if not args.physics_only:
        parse_logic(pipeline, limit=args.limit)
        
    if not args.logic_only:
        parse_physics(pipeline, limit=args.limit)
        
    logger.info("Seeding and Wiring complete.")

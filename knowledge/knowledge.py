"""Knowledge base seeder using ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import networkx as nx

from exact_pipeline.knowledge.retrieval import SearchHit


@dataclass(frozen=True)
class FormulaCard:
    formula_id: str
    family: str
    expression: str
    premise: str
    trigger_terms: Sequence[str]

    def render(self) -> str:
        return f"{self.formula_id}: {self.expression}. {self.premise}"


PHYSICS_FORMULA_CARDS: Sequence[FormulaCard] = (
    FormulaCard(
        "capacitor_energy_voltage",
        "capacitor",
        "E = 0.5*C*U^2",
        "Use SI units: C in farads, U in volts, E in joules.",
        ("capacitor", "energy", "stored", "capacitance", "voltage"),
    ),
    FormulaCard(
        "capacitor_charge",
        "capacitor",
        "Q = C*U",
        "A fully charged capacitor stores charge equal to capacitance times voltage.",
        ("capacitor", "charge", "stores", "fully", "charged"),
    ),
    FormulaCard(
        "capacitance",
        "capacitor",
        "C = Q/U",
        "Capacitance is charge divided by voltage.",
        ("capacitance", "charge", "voltage", "capacitor"),
    ),
    FormulaCard(
        "parallel_plate_capacitance",
        "capacitor",
        "C = epsilon0*epsilon_r*A/d",
        "For parallel plates, convert area and distance to SI before computing.",
        ("parallel", "plate", "area", "distance", "dielectric", "capacitance"),
    ),
    FormulaCard(
        "ohm_law_current",
        "circuit",
        "I = U/R",
        "Ohm's law relates voltage, current, and resistance.",
        ("current", "voltage", "resistance", "ohm"),
    ),
    FormulaCard(
        "ohm_law_voltage",
        "circuit",
        "U = I*R",
        "Voltage equals current times resistance.",
        ("voltage", "current", "resistance", "potential", "difference"),
    ),
    FormulaCard(
        "electric_power",
        "circuit",
        "P = U*I = I^2*R = U^2/R",
        "Pick the electric-power form matching the given quantities.",
        ("power", "current", "voltage", "resistance", "watt"),
    ),
    FormulaCard(
        "resistors_series",
        "circuit",
        "R_eq = sum(R_i)",
        "Ideal series resistances add directly.",
        ("series", "equivalent", "total", "resistance", "resistor"),
    ),
    FormulaCard(
        "resistors_parallel",
        "circuit",
        "1/R_eq = sum(1/R_i)",
        "Ideal parallel resistances add by reciprocals.",
        ("parallel", "equivalent", "total", "resistance", "resistor"),
    ),
    FormulaCard(
        "charge_transport",
        "circuit",
        "Q = I*t",
        "Charge transported by a steady current equals current times time.",
        ("charge", "current", "time", "passes", "transport"),
    ),
    FormulaCard(
        "coulomb_force",
        "electrostatics",
        "F = k*abs(q1*q2)/r^2",
        "Use k = 9e9 N*m^2/C^2 in air/vacuum unless stated otherwise.",
        ("coulomb", "force", "charge", "distance", "separation"),
    ),
    FormulaCard(
        "equal_charge_coulomb",
        "electrostatics",
        "q = sqrt(F*r^2/k)",
        "For two equal charges, invert Coulomb's law.",
        ("equal", "charges", "find", "force", "separated"),
    ),
    FormulaCard(
        "electric_field_point",
        "electrostatics",
        "E = k*abs(q)/r^2",
        "Point-charge electric field uses charge magnitude and distance.",
        ("electric", "field", "strength", "intensity", "point", "charge"),
    ),
    FormulaCard(
        "electric_force_field",
        "electrostatics",
        "F = abs(q)*E",
        "Force on a charge in an electric field is charge magnitude times field strength.",
        ("electric", "field", "force", "charge", "intensity"),
    ),
    FormulaCard(
        "uniform_field_voltage",
        "electrostatics",
        "U = E*d",
        "Uniform field voltage equals field strength times plate separation.",
        ("uniform", "field", "voltage", "distance", "plates"),
    ),
    FormulaCard(
        "resultant_two_forces",
        "vector",
        "R = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))",
        "Use vector addition; same direction adds, opposite subtracts, perpendicular uses Pythagoras.",
        ("resultant", "force", "angle", "same", "opposite", "perpendicular"),
    ),
    FormulaCard(
        "equilateral_triangle_forces",
        "geometry",
        "angle = 60 degrees",
        "The angle between two repulsive forces from vertices of an equilateral triangle is 60 degrees.",
        ("equilateral", "triangle", "angle", "forces", "vertices", "60", "degrees"),
    ),
)


LOGIC_GUIDANCE_CARDS: Sequence[str] = (
    "Translate each natural-language clause into a unary predicate or a Horn rule when possible.",
    "Use ground facts for named students/entities, for example completed_core(Sophia).",
    "Use ForAll(x, A(x) -> B(x)) for universal if-then regulations.",
    "Use And(...) or conjunction when a rule has multiple required conditions.",
    "Answer Yes only when the conclusion follows from the premises; answer No when a negated conclusion follows; otherwise answer Uncertain.",
    "For multiple-choice questions, compare each option against the premise graph and prefer the strongest directly supported option.",
    "Cite premise numbers in the explanation whenever a rule or fact is used.",
)


from exact_pipeline.knowledge.graph_db import HybridDB

def get_physics_knowledge_index(db_path: str, graph_path: str, alpha: float) -> HybridDB:
    return HybridDB(
        collection_name="physics_knowledge",
        items=PHYSICS_FORMULA_CARDS,
        text_fn=lambda card: " ".join(card.trigger_terms) + " " + card.expression + " " + card.premise,
        id_fn=lambda card: card.formula_id,
        db_path=db_path,
        graph_path=graph_path,
        alpha=alpha,
    )


def get_logic_knowledge_index(db_path: str, graph_path: str, alpha: float) -> HybridDB:
    return HybridDB(
        collection_name="logic_knowledge",
        items=LOGIC_GUIDANCE_CARDS,
        text_fn=lambda card: card,
        id_fn=lambda card: f"logic_rule_{hash(card)}",
        db_path=db_path,
        graph_path=graph_path,
        alpha=alpha,
    )


def physics_premises_for_query(query: str, index: HybridDB, max_cards: int = 4) -> List[str]:
    hits = index.search(query, k=max_cards)
    if not hits:
        return [card.render() for card in PHYSICS_FORMULA_CARDS[:max_cards]]
    return [hit.item.render() for hit in hits]


def render_physics_knowledge(query: str, index: HybridDB, max_cards: int = 8) -> str:
    hits = index.search(query, k=max_cards)
    if not hits:
        return "\n".join(card.render() for card in PHYSICS_FORMULA_CARDS[:max_cards])
    return "\n".join(hit.item.render() for hit in hits)


def render_logic_knowledge(query: str, index: HybridDB, max_cards: int = 7) -> str:
    if query:
        hits = index.search(query, k=max_cards)
        if hits:
            return "\n".join(hit.item for hit in hits)
    return "\n".join(LOGIC_GUIDANCE_CARDS[:max_cards])


def get_reasoning_subgraph_context(query: str, index: HybridDB, max_cards: int = 4, max_depth: int = 2) -> List[str]:
    """Retrieves subgraph context for a query."""
    hits, subgraph = index.search_with_subgraph(query, k=max_cards, max_depth=max_depth)
    
    context = []
    if not hits:
        # Fallback to default rendering if no hits
        if "logic" in index.vector_db.collection.name:
            # logic returns a joined string in render_logic_knowledge, so we split it
            return render_logic_knowledge(query, index, max_cards).split("\n")
        else:
            return physics_premises_for_query(query, index, max_cards)
        
    for hit in hits:
        if hasattr(hit.item, "render"):
            context.append(hit.item.render())
        else:
            context.append(str(hit.item))
            
    if subgraph and len(subgraph) > len(hits):
        for node, data in subgraph.nodes(data=True):
            if "text" in data:
                # Omit if already in direct hits to avoid duplication
                is_hit = False
                for hit in hits:
                    if hasattr(hit.item, "formula_id") and hit.item.formula_id == node:
                        is_hit = True
                        break
                    elif hasattr(hit.item, "render") and hit.item.render() == data["text"]:
                        is_hit = True
                        break
                    elif str(hit.item) == data["text"]:
                        is_hit = True
                        break
                
                if not is_hit:
                    context.append(f"Related: {data['text']}")
                    
        # Add connectivity info
        edges = list(subgraph.edges())
        if edges:
            for u, v in edges:
                context.append(f"Dependency: {u} -> {v}")
                
    return context


def merge_premises(*groups: Iterable[str], limit: int = 10) -> List[str]:
    seen = set()
    merged: List[str] = []
    for group in groups:
        for item in group:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
            if len(merged) >= limit:
                return merged
    return merged

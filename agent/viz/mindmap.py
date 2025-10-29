from pathlib import Path
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import networkx as nx

DB_PATH = Path(__file__).resolve().parent.parent.parent / "generated" / "digital_twin.db"
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def draw_mindmap():
    G = nx.DiGraph()
    with engine.begin() as conn:
        tables = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()]
        for tbl in tables:
            G.add_node(tbl, shape='box')
            cols = [r[1] for r in conn.execute(text(f"PRAGMA table_info({tbl});")).fetchall()]
            for col in cols:
                G.add_node(f"{tbl}.{col}", shape='ellipse')
                G.add_edge(tbl, f"{tbl}.{col}")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10)
    plt.title("Mindmap базы данных проекта")
    out_path = Path("generated") / "mindmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"✅  Mindmap сохранена в {out_path}")

if __name__ == "__main__":
    draw_mindmap()
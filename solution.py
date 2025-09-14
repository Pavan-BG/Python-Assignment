
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

try:
    from sqlalchemy import create_engine
    SA_AVAILABLE = True
except Exception:
    SA_AVAILABLE = False
    import sqlite3

import matplotlib.pyplot as plt

try:
    from bokeh.plotting import figure, output_file, save
    BOKEH_AVAILABLE = True
except Exception:
    BOKEH_AVAILABLE = False

class DataShapeError(Exception): ...
class MappingError(Exception): ...

@dataclass
class BaseLoader:
    path: Path
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

@dataclass
class TrainingLoader(BaseLoader):
    def load(self) -> pd.DataFrame:
        df = super().load()
        expected = ['x','y1','y2','y3','y4']
        if list(df.columns) != expected:
            raise DataShapeError(f"Training CSV must have columns {expected}")
        return df.sort_values('x').reset_index(drop=True)

@dataclass
class IdealLoader(BaseLoader):
    def load(self) -> pd.DataFrame:
        df = super().load()
        if df.columns[0] != 'x' or len(df.columns) != 51:
            raise DataShapeError("Ideal CSV must have 51 columns: 'x' + 'y1'..'y50'")
        return df.sort_values('x').reset_index(drop=True)

@dataclass
class TestLoader(BaseLoader):
    def load(self) -> pd.DataFrame:
        df = super().load()
        expected = ['x','y']
        if list(df.columns) != expected:
            raise DataShapeError(f"Test CSV must have columns {expected}")
        return df.sort_values('x').reset_index(drop=True)

@dataclass
class AssignmentSolution:
    train_df: pd.DataFrame
    ideal_df: pd.DataFrame
    test_df: pd.DataFrame

    def compute_sse_matrix(self) -> pd.DataFrame:
        sse = {}
        for tcol in ['y1','y2','y3','y4']:
            diffs = self.ideal_df.drop(columns=['x']).apply(lambda col: (self.train_df[tcol] - col)**2, axis=0)
            sse[tcol] = diffs.sum(axis=0)
        return pd.DataFrame(sse).T

    def select_best_ideals(self) -> Dict[str,str]:
        sse_df = self.compute_sse_matrix()
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(sse_df.values)
            selected_cols = [sse_df.columns[i] for i in col_ind[:4]]
            return {t: selected_cols[i] for i, t in enumerate(sse_df.index)}
        except Exception:
            taken, mapping = set(), {}
            for t in sse_df.index:
                for c in sse_df.loc[t].sort_values().index:
                    if c not in taken:
                        mapping[t] = c; taken.add(c); break
            return mapping

    def max_training_deviation(self, train_col, ideal_col) -> float:
        return float((self.train_df[train_col] - self.ideal_df[ideal_col]).abs().max())

    def map_test_points(self, mapping: Dict[str,str]) -> pd.DataFrame:
        th = {icol: math.sqrt(2.0)*self.max_training_deviation(tcol, icol)
              for tcol, icol in mapping.items()}
        chosen_cols = list(mapping.values())
        merged = pd.merge(self.test_df, self.ideal_df[['x']+chosen_cols], on='x', how='left')
        out = []
        for _, r in merged.iterrows():
            best_col, best_delta = None, None
            for icol in chosen_cols:
                delta = abs(r['y'] - r[icol])
                if delta <= th[icol] and (best_delta is None or delta < best_delta):
                    best_col, best_delta = icol, float(delta)
            if best_col is not None:
                out.append({'x': float(r['x']), 'y': float(r['y']),
                            'delta_y': best_delta, 'ideal_function_no': int(best_col[1:])})
        return pd.DataFrame(out)

    def build_database(self, db_path: Path) -> None:
        if SA_AVAILABLE:
            engine = create_engine(f"sqlite:///{db_path}")
            self.train_df.to_sql('training', engine, if_exists='replace', index=False)
            self.ideal_df.to_sql('ideals', engine, if_exists='replace', index=False)
        else:
            import sqlite3
            conn = sqlite3.connect(db_path)
            self.train_df.to_sql('training', conn, if_exists='replace', index=False)
            self.ideal_df.to_sql('ideals', conn, if_exists='replace', index=False)
            conn.close()

    def write_mapping_table(self, db_path: Path, mapping_df: pd.DataFrame) -> None:
        if SA_AVAILABLE:
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{db_path}")
            mapping_df.to_sql('test_mapping', engine, if_exists='replace', index=False)
        else:
            import sqlite3
            conn = sqlite3.connect(db_path)
            mapping_df.to_sql('test_mapping', conn, if_exists='replace', index=False)
            conn.close()

    def plot_training(self, out_png: Path) -> None:
        plt.figure()
        for col in ['y1','y2','y3','y4']:
            plt.plot(self.train_df['x'], self.train_df[col], label=col)
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Training Functions (y1..y4)")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

    def plot_chosen_ideals(self, mapping: Dict[str,str], out_png: Path) -> None:
        plt.figure()
        for icol in mapping.values():
            plt.plot(self.ideal_df['x'], self.ideal_df[icol], label=icol)
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Chosen Ideal Functions")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

    def plot_test_assignments(self, mapping: Dict[str,str], mapped_df: pd.DataFrame, out_png: Path) -> None:
        plt.figure()
        for icol in mapping.values():
            plt.plot(self.ideal_df['x'], self.ideal_df[icol], label=f"{icol} (ideal)")
        for func_no, grp in mapped_df.groupby('ideal_function_no'):
            plt.scatter(grp['x'], grp['y'], s=15, label=f"test→y{func_no}")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Test Points Mapped to Chosen Ideals")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

    def bokeh_interactive(self, mapping: Dict[str,str], out_html: Path) -> None:
        if not BOKEH_AVAILABLE: return
        p = figure(title="Interactive: Chosen Ideals + Test Assignments",
                   x_axis_label="x", y_axis_label="y", width=900, height=500)
        for icol in mapping.values():
            p.line(self.ideal_df['x'], self.ideal_df[icol], line_width=2)
        p.circle(self.test_df['x'], self.test_df['y'], size=4, alpha=0.6)
        output_file(out_html); save(p)

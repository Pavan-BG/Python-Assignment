
from pathlib import Path
from solution import TrainingLoader, IdealLoader, TestLoader, AssignmentSolution

base = Path(__file__).parent
data = base / "data"

train = TrainingLoader(data / "train.csv").load()
ideal = IdealLoader(data / "ideal.csv").load()
test = TestLoader(data / "test.csv").load()

sol = AssignmentSolution(train, ideal, test)
mapping = sol.select_best_ideals()
mapped = sol.map_test_points(mapping)

db_path = base / "assignment.db"
sol.build_database(db_path)
sol.write_mapping_table(db_path, mapped)

sol.plot_training(base / "plot_training.png")
sol.plot_chosen_ideals(mapping, base / "plot_chosen_ideals.png")
sol.plot_test_assignments(mapping, mapped, base / "plot_test_assignments.png")
sol.bokeh_interactive(mapping, base / "interactive.html")

print("Mapping:", mapping)
print("Mapped rows:", len(mapped))

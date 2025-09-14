
import pandas as pd
from solution import AssignmentSolution

def test_sse_matrix_shapes():
    x = list(range(5))
    train = pd.DataFrame({"x":x,"y1":[0,1,2,3,4],"y2":[1,2,3,4,5],"y3":[2,3,4,5,6],"y4":[3,4,5,6,7]})
    ideal = pd.DataFrame({"x":x, **{f"y{i}":[i-1+j for j in range(5)] for i in range(1,51)}})
    test = pd.DataFrame({"x":[0,1],"y":[0,1]})
    s = AssignmentSolution(train, ideal, test)
    sse = s.compute_sse_matrix()
    assert sse.shape == (4,50)

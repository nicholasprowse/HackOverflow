import sys
from rubik.solve import Solver 
from rubik.cube import Cube
c = Cube(sys.argv[1])
solver = Solver(c)
solver.solve()
print(solver.moves)

import sys
from rubik.solve import Solver 
from rubik.cube import Cube
c = Cube(sys.argv[1])
solver = Solver(c)
solver.solve()
output = solver.moves
print(' '.join(output))
f = open("steps.txt","w")
f.write(' '.join(output))
f.close()


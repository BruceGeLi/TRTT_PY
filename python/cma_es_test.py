import cma

es = cma.CMAEvolutionStrategy([0, 0.1, 0.2], 0.1)
while not es.stop():
    solutions = es.ask()
    #print(solutions)
    es.tell(solutions, [cma.ff.rosen(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()
    

es.result_pretty()
cma.plot()
import pygmo as pg


class Algorithms:
    """
    This class  holds a compilation of the different optimization algorithms that can be used to solve the trajectory
    problem. One can call this class to run different optimization algorithms for the same problem to see which
    algorithm does the best.
    """

    def __init__(self, problem):
        """
        Defining the initial user defined problem (udp).

        Args:
            - problem (``pykep udp class``): The pykep problem class that that optimization algorithm will be solving.
        """

        self.problem = problem
        self.names = [self.self_adaptive_differential_algorithm] #storing the function names of the different algorithms

    def solve(self, algo):
        """
        Solving the pygmo problem of whatever algorithm its given and printing + plotting the result.

        Args:
            - algo (``pygmo algorithm``): The algorithm that will be used to solve the pygmo problem.
        """

        algo.evolve(10)
        algo.wait()
        sols = algo.get_champions_f()
        sols2 = [item[0] for item in sols]
        idx = sols2.index(min(sols2))
        print("Done!!") #Solutions found are: ", algo.get_champions_f())
    
        return algo.get_champions_x()[idx]

    def self_adaptive_differential_algorithm(self, generations=100, islands=8, island_population=20):
        """
        Defines a self adaptive differential algorithm on different islands.

        Args:
            - generations (``int``): The number of generations the algorithm will run.
            - islands (``int``): The number of parallel islands to run on.
            - island_population (``int``): The number of cases run per island.
        """
        uda = pg.sade(gen=generations)
        archi = pg.archipelago(algo=uda, prob=self.problem, n=islands, pop_size=island_population)
        print(
            "Running a Self-Adaptive Differential Evolution Algorithm .... on {} parallel islands".format(islands))

        return self.solve(archi)
    
    def extended_ant_colony(self, generations=1000, kernel=15, q_convergence=1.0, oracle_penalty=1e9, accuracy=0.01, thresh=500, std_conv=7, islands=8, island_population=20):
        uda=pg.gaco(gen=generations, ker=kernel, q=q_convergence, oracle=oracle_penalty, acc=accuracy, threshold=thresh, n_gen_mark=std_conv)
        archi = pg.archipelago(algo=uda, prob=self.problem, n=islands, pop_size=island_population)
        print(
            "Running an Extended Ant-Colony Algorithm .... on {} parallel islands".format(islands))

        return self.solve(archi)

    def func_names(self):
        """
        Returns the names of the different functions that have been implemented in this class as an array.
        """
        return self.names

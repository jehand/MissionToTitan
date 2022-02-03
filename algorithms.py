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
        self.local_algo = [self.self_adaptive_differential_algorithm, self.extended_ant_colony,
                           self.particle_swarming_optimization, self.simple_genetic_algorithm, self.calculus]  # storing the function names of the different algorithms
        self.global_algo = [self.monotonic_basin_hopping]

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
        Defines a self adaptive differential algorithm.

        Args:
            - generations (``int``): The number of generations the algorithm will run.
        """
        uda = pg.sade(gen=generations)
        print(
            "Using a Self Adaptive Differential Optimization Algorithm ....")

        return uda
    
    def extended_ant_colony(self, generations=1000, kernel=15, q_convergence=1.0, oracle_penalty=1e9, accuracy=0.01,
                            thresh=500, std_conv=7):
        """
        Defines an extended ant colony algorithm.

        Args:
            - generations (``int``): The number of generations the algorithm will run.
            - kernel (``int``): The number of solutions stored in the solution archive.
            - q_convergence (``float``): Convergence speed parameter (the smaller, the faster)
            - oracle_penalty (``float``): Oracle parameter used in the penalty method.
            - accuracy (``float``): Parameter for maintaining a minimum penalty function's values distances.
            - thresh (``int``): When the generations reach the threshold, q_convergence is set to 0.01 automatically.
            - std_conv (``float``): Determines the convergence speed of the standard deviations values.
        """
        uda = pg.gaco(gen=generations, ker=kernel, q=q_convergence, oracle=oracle_penalty, acc=accuracy,
                      threshold=thresh, n_gen_mark=std_conv)
        print(
            "Using an Extended Ant Colony Optimization Algorithm ....")

        return uda

    def simple_genetic_algorithm(self, generations=100):
        """
        Defines a simple genetic algorithm.

        Args:
            - generations (``int``): The number of generations the algorithm will run.
        """
        uda = pg.sga(gen=generations)
        print(
            "Using a Simple Genetic Algorithm ....")

        return uda

    def particle_swarming_optimization(self, generations=100):
        """
        Defines a particle swarm optimization algorithm.

        Args:
            - generations (``int``): The number of generations the algorithm will run.
        """
        uda = pg.pso(gen=generations)
        print(
            "Using a Particle Swarming Optimization Algorithm ....")

        return uda

    def monotonic_basin_hopping(self, algo, stopping=5, perturbation=0.05):
        """
        Defines a particle swarm optimization on different islands.

        Args:
            - algo (``pygmo.algorithm``): The local optimization function.
            - stopping (``int``): Number of no improvements before halting optimization.
            - perturbation (``float``): Perturbation width.
        """
        uda = pg.mbh(algo=algo, stop=stopping, perturb=perturbation)
        print(
            "Running a Monotonic Basin Hopping Algorithm ....")

        return uda

    def archipelago(self, uda, islands=8, island_population=20):
        """
        Defines an archipelago.

        Args:
            - uda (``pygmo.algorithm``): A pygmo algorithm.
            - islands (``int``): The number of parallel islands to run on.
            - island_population (``int``): The number of cases run per island.
        """
        archi = pg.archipelago(algo=uda, prob=self.problem, n=islands, pop_size=island_population)
        print(
            "Using Archipelago .... on {} parallel islands".format(islands))
        return self.solve(archi)
    
    def calculus(self, algo="slsqp"):
        """
        Defines a calculus based approach to optimizing using nlopt.

        Args:
            - algo (``str``): Name of the calculus based approach in nlopt.
        """
        uda = pg.nlopt(algo)
        print(
            "Running a Calculus Based " + algo + " Approach ....")
        
        return uda

    def local_opt_names(self):
        """
        Returns the names of the different local optimization functions that have been implemented in this class as an
        array.
        """
        return self.local_algo

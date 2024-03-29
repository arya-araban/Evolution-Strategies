# Evolution-Strategies

Implementation of the three main algorithms for Evolution Strategies. The [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) benchmark function has been used to test each of the methods.

 **Classic ES:**  In each generation the population size is always equal to only one. If the child scores better than its parent in a certain generation, then it'll be selected as the parent for the next generation. 
 
 **ES(μ, λ):** In each generation we select μ best parents, and λ is the size of the population.In this version of ES, each selected parent has λ\μ children, which these children becomes the population for the next generation. 
 
 **ES(μ + λ):** This method is similar to ES(μ, λ), with the only difference being that for the next generation, the population is chosen between the children AND their parents.

# MVA-RL-project (Pierre Bizeul, Jules Kozolinsky)
RL project from the RL course of [Alessandro Lazaric](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html).

Supervised by [Ronan Fruit](https://ronan.fruit.nom.fr/).


## Thompson Sampling (PSRL) under bias span constraint
Topic:  Reinforcement Learning, Exploration-Exploitation, Planning in Markov Decision Processes

Description:  The Exploration-Exploitation trade-off is a fundamental dilemma in on-line reinforcement learning. Algorithms addressing this dilemma with theoretical performance guarantees have been proposed for the discounted, average and finite horizon settings. In the finite horizon setting, the usual criterion of performance is the notion of “regret” as in MAB. One of the current state-of-the-art algorithms PSRL [1] has a regret scaling linearly with the “diameter” of the unknown MDP in the worst-case (the diameter being a measure of how easy it is to navigate between any two states of the MDP). Although it has been shown that the dependency in the diameter is unavoidable in the worst-case, when additional properties on the optimal policy are known beforehand, the regret can, in theory, be drastically improved [2]. Unfortunately, exploiting this additional prior knowledge on the optimal policy requires solving an optimization problem (namely, “planning under bias span constraint”) and no algorithm has been derived so far to solve it. We recently started investigating a new Bellman operator converging to the solution of this problem for many MDPs. In this research project, the student(s) is/are expected to:

1. Implement PSRL,
2. Integrate the modified Bellman operator to PSRL,
3. Compare the empirical regret of the two

[1] Osband Ian, Van Roy Benjamin, and Russo Daniel. (More) efficient reinforcement learning via posterior sampling. In Proceedings of the 26th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'13), Vol. 2. Curran Associates Inc., USA, 3003-3011, 2013.

[2] Peter L. Bartlett and Ambuj Tewari. REGAL: a regularization based algorithm for reinforcement learning in weakly communicating MDPs. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI '09). AUAI Press, Arlington, Virginia, United States, 35-42, 2009.

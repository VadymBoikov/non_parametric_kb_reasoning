

### Implementation for paper:
#### [Non-Parametric Reasoning on Knowledge Bases](https://www.semanticscholar.org/paper/Non-Parametric-Reasoning-on-Knowledge-Bases-Gates-Ambani/1d478cefc11e162fec7428a7315487de87adbf1f) 


### To run:
1. Install libraries `pip install -r requirements.txt` (python 3.7)
2. Unpack data `sh preprocess.sh`
3. Run `main.py --dataset=WN18RR` 


### Achieved results:

|Dataset   | Hits@1 | Hits@3  | Hits@10 | MRR   |
|---       | ---    | ---     | ---     | ---   |
| WN18RR   | 0.38   | 0.4     | 0.43    |  0.4  |
| FK15K-237|        |         |         |       |


#### P.S. running time:

1. WN18RR ~1.5min
2. FK15K ~?


Most time is used for memory creation. (cases and similarity matrix)
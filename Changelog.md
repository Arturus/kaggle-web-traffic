2018-10-15
- Model updated to work with a modern Tensorflow (>=1.10)
- Switched to Adam instead of COCOB (COCOB don't works with TF > 1.4)
- No parameter tuning for Adam performed, therefore model probably has
 suboptimal training rate and did'nt reproduce exact result from the competition
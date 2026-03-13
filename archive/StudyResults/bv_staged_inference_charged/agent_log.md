# Staged Inference Study - Agent Log

## Approach
- Stage 1: Alpha inference with k0 fixed at true values
- Stage 2: k0 inference with alpha fixed at Stage 1 result
- Stage 3: Joint refinement from Stage 1+2 warm start
- Stage 4: Direct joint inference (comparison baseline)

## Results
### stage1
- k0: [0.0012631578947368421, 5.2631578947368424e-05]
- alpha: [0.6289414394553389, 0.1646634075167531]
- loss: 3.331051e-05
- time: 157.4s

### stage2
- k0: [0.0013256745339510418, 1.6371198960538748e-06]
- alpha: [0.6289414394553389, 0.1646634075167531]
- loss: 3.278108e-05
- time: 988.1s

### stage3
- k0: [0.001505642168035109, 1.6379251893581279e-06]
- alpha: [0.590707800600389, 0.16541813475605688]
- loss: 3.210947e-05
- time: 112.4s

### stage4
- k0: [0.004058378767305928, 0.00019483243165419091]
- alpha: [0.15430155544137045, 0.95]
- loss: 1.900193e-05
- time: 346.9s

## Conclusion
Staged time: 1258s vs Direct: 347s

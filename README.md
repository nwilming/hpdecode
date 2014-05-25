hpdecode
========

This script is a supplement to the Action and Cognition 2 lecture at the
University of Osnabrueck (2014). It demonstrates two decoding approaches
for place fields from the hippocampus. Both approaches (direct basis) and
the probabilistic approach are taken from:

Interpreting Neuronal Population Activity by Reconstruction: Unified Framework 
With Application to Hippocampal Place Cells
Kechen Zhang , Iris Ginzburg , Bruce L. McNaughton , Terrence J. Sejnowski
Journal of NeurophysiologyPublished 1 February 1998Vol. 79no. 1017-1044


This is how to get started:

```python
import reconstruct as rc
pfields, arena, aX, aY, centers = rc.setup()
rates, spikes = rc.simulate_spikes(pfields, 5,5)
posterior = rc.decode_bayes(pfields, spikes, arena)
activity = rc.decode_directbasis(pfields, spikes, arena)
```

And now posterior and activity contain the estimates for locations in the
discretized area, so you might want to start by plotting the two.


Interesting points to investigate:
    - Dependence of decoding strategy on firing rate of place fields
    - What happens if we have non poisson neurons?
    - Look at different arena geometries.
    - What happens when the place fields are not nicely Gaussian anymore?
    - What if they are not even continuous in space anymore?


What AstraCell currently proves:
- It can simulate a 32-cell pack.
- It can inject physical and sensor faults.
- It can compute CRLB/Fisher-based detectability.
- It can mark faults as diagnosable, weak, or unobservable.
- It can recommend a sensor or stronger excitation when a fault is not observable.

What AstraCell does NOT prove:
- That the model matches real EV batteries.
- That the CRLB bounds hold under correlated real-world noise.
- That current is perfectly known.
- That a single-node thermal model is realistic.
- That real BMS data exposes enough telemetry.


Important for post-project evaluation
**    I initially claimed correlated noise always made AstraCell more conservative. A test falsified that. The corrected AR(1) whitening showed that correlated noise reallocates information: it hurts DC-like capacity signatures but can strengthen pulsed R0/hA signatures. That changed the recommendation from “add a thermocouple” to “run a pulse test” in one case **
# Iteration 600 evidence

Long TP4 M128 replay-interleaved cold-L2 confirmation. Each table row contains 300 independently cold samples for the candidate and 300 for its same-window selected multi-kernel control. Cache clear is a separate excluded 256 MiB operation.

| Window | Variant | Candidate median ms | Same-window multi median ms | Candidate / control |
|---|---:|---:|---:|---:|
| OFF_A | inline, 8 CTA/SM | 0.3738399893 | 0.3280640095 | 1.1395336837 |
| ON_A | compact outline, 9 CTA/SM | 0.3675680012 | 0.3238240033 | 1.1350857174 |
| ON_B | compact outline, 9 CTA/SM | 0.3668799996 | 0.3228960037 | 1.1362172198 |
| OFF_B | inline, 8 CTA/SM | 0.3766559958 | 0.3303840011 | 1.1400551920 |

The controls moved by about 1.8% between process groups, so raw candidate medians are not compared across processes. Mean normalized ON/OFF is 0.9963651610, a 0.3635% candidate improvement after cancelling that drift. The direction agrees with Iter599's independent 0.5508% short bracket.

All four windows used the same 248-active-expert route realization, 768 routed rows, and 1,992 padded rows. Candidate/control numerical metrics were identical in every window.


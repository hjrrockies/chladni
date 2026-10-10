| case | ref levels (trusted) | lappy_fresh | lappy_frozen | chladni_1e-08 | chladni_1e-10 |
|---|---|---|---|---|---|
| T1.1-bad | 9 (9) | **SILENT** drop 1, extr 1 | **SILENT** drop 1, extr 1 | ok (9 unmet) | ok (9 unmet) |
| T1.1-bisection | 11 (11) | **SILENT** drop 1, extr 1 | **SILENT** drop 1, extr 1 | ok (10 unmet) | ok (11 unmet) |
| T1.1-good | 11 (11) | ok | ok | ok (10 unmet) | ok (11 unmet) |
| T1.2 | 10 (10) | ok | **SILENT** drop 1, extr 1 | ok (1 unmet) · UNSTABLE (2 outcomes, 1/5 wrong) | silent (unmet only) drop 2 · UNSTABLE (2 outcomes, 2/5 wrong) |
| T1.3 | 9 (9) | ok | ok | ok (9 unmet) | ok (9 unmet) · UNSTABLE (2 outcomes, 0/5 wrong) |
| T1.4-hex | 10 (10) | ok |  | ok | ok (1 unmet) |
| T1.4-oct | 9 (9) | ok |  | ok · UNSTABLE (2 outcomes, 3/5 wrong) | ok |
| T1.5-r1.633 | 15 (15) | ok |  | ok | ok |
| T1.5-r2 | 20 (20) | ok |  | ok | silent (unmet only) drop 1, extr 1 · UNSTABLE (3 outcomes, 3/5 wrong) |
| T2.1-pass1 | 10 (8) | unscored |  | ok (9 unmet) | ok (9 unmet) |
| T2.1-rerun | 10 (9) | unscored |  | ok (10 unmet) | ok (10 unmet) |
| T2.2 | 11 (11) | **SILENT** drop 4, extr 4 | **SILENT** drop 4, extr 4 | silent (unmet only) drop 1, extr 1 · UNSTABLE (2 outcomes, 5/5 wrong) | silent (unmet only) drop 1 |
| T2.3-endpoint | 11 (11) | ok | **SILENT** drop 2, extr 2 | ok (10 unmet) | ok (11 unmet) |
| T2.3-pre-ref0 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (11 unmet) | ok (11 unmet) |
| T2.3-pre-ref3 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (11 unmet) | ok (11 unmet) |
| T2.4-N5-L4 | 10 (10) | ok |  | ok (1 unmet) | ok (1 unmet) |
| T2.4-N5-L5 | 8 (7) | unscored |  | unscored | unscored |
| T2.4-N5-L28 | 10 (10) | ok |  | ok (1 unmet) | ok (2 unmet) |
| T2.4-N4-L25 | 8 (7) | ok |  | ok (5 unmet) | ok (8 unmet) |
| T2.5 | 9 (9) | **SILENT** drop 5, extr 5 | **SILENT** drop 5, extr 5 | ok (9 unmet) | ok (9 unmet) |
| T2.6-N6k3-L4-ref0 | 10 (9) | **SILENT** drop 2, extr 2 | ok | ok (1 unmet) | ok (5 unmet) |
| T2.6-N5k5run3-ref0 | 14 (9) | unscored | unscored | unscored | unscored |
| T2.6-N5k4-L7-ref0 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | count-flagged drop 1 | ok (11 unmet) |
| T2.6-N5k4-L7-ref3 | 10 (9) | unscored | unscored | ok (9 unmet) | ok (9 unmet) |
| T2.6-N3k2-L8-ref0 | 17 (0) | unscored | unscored | unscored | unscored |
| T2.6-N3k5-L6-ref0 | 11 (6) | unscored | unscored | ok (7 unmet) | ok (7 unmet) |
| T2.6-N3k4-L8-ref0 | 10 (10) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (10 unmet) | ok (10 unmet) |
| T2.7-lowest | 14 (11) | unscored | unscored | unscored | unscored |
| T2.7-median | 14 (11) | **SILENT** drop 3, extr 3 | **SILENT** drop 3, extr 3 | ok (1 unmet) | ok (1 unmet) |
| T2.7-highest | 14 (10) | unscored | unscored | unscored | unscored |
| T2.8-N5s2 | 8 (8) | ok | ok | ok (6 unmet) | ok (8 unmet) |
| T2.8-N6s1 | 9 (3) | unscored | unscored | unscored | unscored |
| T2.8-kink | 11 (9) | ok | exception | ok | ok |
| T2.9-seed | 9 (9) | ok |  | ok (7 unmet) | ok (9 unmet) |
| T2.9-N4opt | 8 (7) | unscored |  | ok (7 unmet) | ok (8 unmet) |
| T2.9-N5k3s5 | 10 (10) | ok |  | ok (1 unmet) | ok (1 unmet) |
| T2.9-ngon5 | 6 (6) | ok | ok | ok | ok |
| T2.9-ngon8 | 4 (3) | unscored | unscored | ok · UNSTABLE (2 outcomes, 1/5 wrong) | ok |
| T2.9-ngon12 | 4 (3) | ok | ok | count-flagged drop 2 | silent (unmet only) drop 4 · UNSTABLE (3 outcomes, 3/5 wrong) |
| T2.9-ngon16 | 2 (1) | unscored | unscored | ok | ok (2 unmet) |
| T2.9-N5k3convex | 10 (10) | **SILENT** drop 5, extr 5 | **SILENT** drop 5, extr 5 | ok (5 unmet) | ok (10 unmet) · UNSTABLE (2 outcomes, 1/5 wrong) |

* `lappy_fresh`: {'wrong-silent': 11, 'ok': 17, 'unscored': 13}
* `lappy_frozen`: {'wrong-silent': 12, 'ok': 6, 'unscored': 9, 'exception': 1}
* `chladni_1e-08`: {'ok': 32, 'wrong-silent': 1, 'unscored': 6, 'wrong': 2}; stable under window jitter on 31 of 35 scored cases; median wall per solve 5.4 s
* `chladni_1e-10`: {'ok': 31, 'wrong-silent': 4, 'unscored': 6}; stable under window jitter on 30 of 35 scored cases; median wall per solve 5.5 s

| case | ref levels (trusted) | lappy_fresh | lappy_frozen | chladni_1e-08 | chladni_1e-13 |
|---|---|---|---|---|---|
| T1.1-bad | 9 (9) | **SILENT** drop 1, extr 1 | **SILENT** drop 1, extr 1 | ok (9 unmet) | ok (9 unmet) |
| T1.1-bisection | 11 (11) | **SILENT** drop 1, extr 1 | **SILENT** drop 1, extr 1 | ok (10 unmet) | ok (11 unmet) |
| T1.1-good | 11 (11) | ok | ok | ok (10 unmet) | ok (11 unmet) |
| T1.2 | 10 (9) | unscored | unscored | ok (1 unmet) | ok (9 unmet) |
| T1.3 | 9 (9) | ok | ok | ok (9 unmet) | ok (9 unmet) |
| T1.4-hex | 10 (10) | ok |  | ok | silent (unmet only) drop 10 |
| T1.4-oct | 9 (9) | ok |  | **SILENT** drop 2 | silent (unmet only) drop 10, extr 1 |
| T1.5-r1.633 | 15 (15) | ok |  | ok | silent (unmet only) extr 2 |
| T1.5-r2 | 20 (20) | ok |  | ok | count-flagged drop 8, extr 3, mult 5 |
| T2.1-pass1 | 10 (8) | ok |  | ok (9 unmet) | ok (10 unmet) |
| T2.1-rerun | 10 (9) | unscored |  | ok (10 unmet) | ok (10 unmet) |
| T2.2 | 11 (11) | **SILENT** drop 4, extr 4 | **SILENT** drop 4, extr 4 | silent (unmet only) drop 1 | silent (unmet only) drop 1 |
| T2.3-endpoint | 11 (11) | ok | **SILENT** drop 2, extr 2 | ok (10 unmet) | ok (11 unmet) |
| T2.3-pre-ref0 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (11 unmet) | ok (11 unmet) |
| T2.3-pre-ref3 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (11 unmet) | ok (11 unmet) |
| T2.4-N5-L4 | 10 (10) | ok |  | ok (1 unmet) | ok (10 unmet) |
| T2.4-N5-L5 | 8 (7) | unscored |  | unscored | unscored |
| T2.4-N5-L28 | 10 (10) | ok |  | ok (1 unmet) | ok (10 unmet) |
| T2.4-N4-L25 | 8 (7) | ok |  | ok (5 unmet) | ok (8 unmet) |
| T2.5 | 9 (9) | **SILENT** drop 5, extr 5 | **SILENT** drop 5, extr 5 | ok (9 unmet) | ok (1 unc, 8 unmet) |
| T2.6-N6k3-L4-ref0 | 10 (9) | unscored | unscored | ok (1 unmet) | ok (10 unmet) |
| T2.6-N5k5run3-ref0 | 13 (11) | unscored | unscored | unscored | unscored |
| T2.6-N5k4-L7-ref0 | 11 (11) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | count-flagged drop 1 | ok (11 unmet) |
| T2.6-N5k4-L7-ref3 | 11 (11) | ok | **SILENT** drop 2, extr 2 | count-flagged drop 2 | silent (unmet only) drop 2 |
| T2.6-N3k2-L8-ref0 | 15 (1) | unscored | unscored | unscored | unscored |
| T2.6-N3k5-L6-ref0 | 11 (6) | unscored | unscored | ok (7 unmet) | ok (7 unmet) |
| T2.6-N3k4-L8-ref0 | 10 (10) | **SILENT** drop 2, extr 2 | **SILENT** drop 2, extr 2 | ok (10 unmet) | ok (10 unmet) |
| T2.7-lowest | 13 (10) | unscored | unscored | unscored | unscored |
| T2.7-median | 13 (11) | **SILENT** drop 3, extr 3 | **SILENT** drop 3, extr 3 | ok (1 unmet) | ok (13 unmet) |
| T2.7-highest | 14 (11) | unscored | unscored | unscored | unscored |
| T2.8-N5s2 | 8 (8) | ok | ok | ok (6 unmet) | ok (1 unc, 7 unmet) |
| T2.8-N6s1 | 9 (4) | unscored | unscored | unscored | declined |
| T2.8-kink | 10 (10) | ok | exception | ok | silent (unmet only) extr 2 |
| T2.9-seed | 9 (9) | ok |  | ok (7 unmet) | ok (9 unmet) |
| T2.9-N4opt | 8 (7) | unscored |  | ok (7 unmet) | ok (8 unmet) |
| T2.9-N5k3s5 | 10 (10) | ok |  | ok (1 unmet) | ok (10 unmet) |
| T2.9-ngon5 | 6 (6) | ok | ok | ok | silent (unmet only) drop 6 |
| T2.9-ngon8 | 5 (5) | ok | ok | **SILENT** drop 2 | silent (unmet only) drop 6 |
| T2.9-ngon12 | 3 (1) | unscored | unscored | ok (1 unmet) | ok (3 unmet) |
| T2.9-ngon16 | 2 (0) | unscored | unscored | unscored | unscored |
| T2.9-N5k3convex | 13 (7) | **SILENT** drop 5, extr 5 | **SILENT** drop 5, extr 5 | ok (5 unmet) | ok (10 unmet) |

* `lappy_fresh`: {'wrong-silent': 10, 'ok': 18, 'unscored': 13}
* `lappy_frozen`: {'wrong-silent': 12, 'ok': 5, 'unscored': 10, 'exception': 1}
* `chladni_1e-08`: {'ok': 29, 'wrong-silent': 3, 'unscored': 7, 'wrong': 2}
* `chladni_1e-13`: {'ok': 25, 'wrong-silent': 8, 'wrong': 1, 'unscored': 6, 'declined': 1}

# UNSW-NB15 Attack-Category Reference

A working reference for the ten classes in the UNSW-NB15 dataset (normal
traffic plus nine attack categories). It supports the EDA, the rule-based
IDS design (Part 6.1), and the RQ1 feature-group analysis (Part 10.1).

Record counts below are from the **training partition** (175,341 records);
proportions are similar in the pooled 257,673-record dataset.

| Category | Train records | Description | Typical network behaviour | Candidate discriminative features | Expected difficulty |
|---|---:|---|---|---|---|
| Normal | 56,000 | Legitimate background traffic. | Regular, well-formed flows. | (baseline) | — |
| Generic | 40,000 | Attacks effective against a cipher/protocol regardless of its configuration. | Varied; often distinctive packet-loss patterns. | dloss, sloss, sbytes, dbytes | Low |
| Exploits | 33,393 | Use of known vulnerabilities in software or protocols. | Crafted packet sequences targeting a service. | ct_state_ttl, ct_dst_ltm, is_sm_ips_ports | Medium |
| Fuzzers | 18,184 | Feeding malformed or random input to crash a service. | High packet rate, many short connections. | dur, rate, ct_state_ttl | Medium |
| DoS | 12,264 | Denial of service — resource exhaustion. | High-volume, high-rate traffic. | rate, sload, dload, sbytes, dbytes | Medium |
| Reconnaissance | 10,491 | Information gathering / probing before an attack. | Many targets, low volume per target. | ct_dst_sport_ltm, ct_src_dport_ltm | Medium |
| Analysis | 2,000 | Intrusion analysis — port scans, spam, HTML penetrations. | Probing across ports and services. | sttl, ct_dst_sport_ltm | High (rare) |
| Backdoor | 1,746 | Bypassing authentication for stealthy remote access. | Persistent, low-volume connections. | synack, ackdat, tcprtt, dmean | High (rare, stealthy) |
| Shellcode | 1,133 | A small code payload used to exploit a vulnerability. | Specific payload patterns; rare. | ct_srv_src, ct_srv_dst, swin, dwin | Very high (rare) |
| Worms | 130 | Self-replicating malware spreading across hosts. | Rapid propagation to many destinations. | ct_dst_sport_ltm, is_sm_ips_ports | Very high (extremely rare) |

## Notes for the analysis

- **Rarity drives difficulty.** Worms (130), Shellcode (1,133), Backdoor
  (1,746) and Analysis (2,000) have very few examples. Per-class recall
  on these classes is the core subject of research question RQ2.
- **Structural similarity drives confusion.** Analysis, Reconnaissance,
  and Backdoor all involve probing or low-volume behaviour and are easy
  for a model to mix up; the EDA attack-pattern figure shows where the
  signatures overlap.
- **Candidate features** listed above are starting hypotheses for the
  rule-based IDS and the RQ1 feature-group study — they are refined
  empirically against the data, not assumed.
